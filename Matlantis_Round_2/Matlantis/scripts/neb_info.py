# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 18:23:42 2026

@author: karlr
"""

import time

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from ase.optimize import FIRE, BFGS
from ase.geometry import find_mic
from matplotlib import pyplot as plt


FREE_ENERGY = False


def set_neb_info(images, set_velocities=False):
    """Sets info for the neb."""
    # The first image's tangent is just towards the next image
    fwd_direction, _ = find_mic(images[1].get_positions() -
                                images[0].get_positions(),
                                images[0].cell, images[0].pbc)
    images[0].info['neb_distance'] = 0.0
    images[0].info['neb_angle'] = np.nan
    images[0].info['neb_tangent'] = fwd_direction / \
        np.linalg.norm(fwd_direction, ord='fro')
    try:
        images[0].info['neb_force'] = np.sum(images[0].get_forces() *
                                             images[0].info['neb_tangent'])
    except RuntimeError:
        images[0].info['neb_force'] = np.nan

    for prev_img, img, next_img in zip(images[:-2], images[1:-1], images[2:]):
        fwd_direction, _ = find_mic(next_img.get_positions() -
                                    img.get_positions(),
                                    img.cell, img.pbc)
        rev_direction, _ = find_mic(img.get_positions() -
                                    prev_img.get_positions(),
                                    img.cell, img.pbc)

        energies = [prev_img.get_potential_energy(FREE_ENERGY),
                    img.get_potential_energy(FREE_ENERGY),
                    next_img.get_potential_energy(FREE_ENERGY)]

        energy_max = max(abs(energies[2] - energies[1]),
                         abs(energies[0] - energies[1]))
        energy_min = max(abs(energies[2] - energies[1]),
                         abs(energies[0] - energies[1]))

        if energies[0] < energies[1] < energies[2]:
            tangent = fwd_direction
        elif energies[0] > energies[1] > energies[2]:
            tangent = rev_direction
        elif energies[0] < energies[2]:
            tangent = fwd_direction * energy_max + rev_direction * energy_min
        elif energies[0] > energies[2]:
            tangent = fwd_direction * energy_max + rev_direction * energy_min
        else:
            tangent = fwd_direction + rev_direction

        img.info['neb_distance'] = prev_img.info['neb_distance'] + \
            np.linalg.norm(rev_direction, ord='fro')
        img.info['neb_tangent'] = tangent / np.linalg.norm(tangent, ord='fro')

        try:
            img.info['neb_force'] = np.sum(img.get_forces() *
                                           img.info['neb_tangent'])
        except RuntimeError:
            img.info['neb_force'] = np.nan

        cos_ang = np.sum(-fwd_direction * rev_direction) / \
            (np.linalg.norm(fwd_direction, ord='fro') *
             np.linalg.norm(rev_direction, ord='fro'))

        img.info['neb_angle'] = np.rad2deg(np.arccos(cos_ang))

    rev_direction, _ = find_mic(images[-2].get_positions() -
                                images[-1].get_positions(),
                                images[-1].cell, images[-1].pbc)
    images[-1].info['neb_distance'] = images[-2].info['neb_distance'] + \
        np.linalg.norm(rev_direction, ord='fro')
    images[-1].info['neb_angle'] = np.nan
    images[-1].info['neb_tangent'] = -rev_direction / \
        np.linalg.norm(rev_direction, ord='fro')
    try:
        images[-1].info['neb_force'] = np.sum(images[-1].get_forces() *
                                              images[-1].info['neb_tangent'])
    except RuntimeError:
        images[-1].info['neb_force'] = np.nan

    if set_velocities:
        for img in images:
            img.set_velocities(img.info['neb_tangent'])


def print_neb(images):
    """Prints the neb info for the atoms."""
    print('  i',
          'Dist [Å]',
          'Ang  [°]',
          'Abs energy  [eV]',
          'Rel energy  [eV]',
          'Par force [eV/Å]',
          sep='  ')

    try:
        energy = images[0].get_potential_energy(FREE_ENERGY)
    except RuntimeError:
        energy = 0.0

    for i, img in enumerate(images):
        try:
            img_energy = img.get_potential_energy(FREE_ENERGY)
        except RuntimeError:
            img_energy = np.nan

        print(f'{i: >3d}',
              f'{img.info["neb_distance"]: 8.2f}',
              f'{img.info["neb_angle"]: 8.2f}',
              f'{img_energy: 16.4f}',
              f'{img_energy - energy: 16.6f}',
              f'{img.info["neb_force"]: 16.6f}',
              sep='  ')


def plot_neb(images, ax=None, color='b', marker_color=None,
             show_force=True, scale_distance=False):
    """Plot the neb."""
    neb_distances = np.array([img.info['neb_distance'] for img in images])
    neb_enegies = np.array([img.get_potential_energy(FREE_ENERGY)
                            for img in images])
    neb_forces = np.array([img.info['neb_force'] for img in images])

    neb_enegies = neb_enegies

    neb_poly = CubicHermiteSpline(neb_distances, neb_enegies, -neb_forces)

    distance = np.linspace(neb_distances[0], neb_distances[-1], num=100)
    energy = neb_poly(distance)

    # Find the maximum of the neb
    # barrier_index = np.argmax(energy)
    # barrier_distance = distance[barrier_index]
    # barrier_energy = energy[barrier_index]

    # print(f'Forward barrier: {barrier_energy - neb_enegies[0]:16.6f}')
    # print(f'Reverse barrier: {barrier_energy - neb_enegies[-1]:16.6f}')
    # print(f'Reaction energy: {neb_enegies[-1] - neb_enegies[0]:16.6f}')

    if marker_color is None:
        marker_color = color

    if ax is None:
        fig, ax = plt.subplots()

    if scale_distance:
        ax.plot(distance/neb_distances[-1], energy, '-', color=color)
    else:
        ax.plot(distance, energy, '-', color=color)

    if show_force and not scale_distance:
        dx = (neb_distances[-1] - neb_distances[0]) / (len(neb_distances) - 1)

        for x, y, dydx in zip(neb_distances, neb_enegies, -neb_forces):
            ax.plot([x - dx/2, x + dx/2], [y - dx*dydx/2, y + dx*dydx/2], 'k-')

    if scale_distance:
        ax.plot(neb_distances/neb_distances[-1], neb_enegies, '.', color=marker_color)
    else:
        ax.plot(neb_distances, neb_enegies, '.', color=marker_color)

    '''
    ax.annotate('',
                (neb_distances[0], barrier_energy),
                (neb_distances[0], neb_enegies[0]),
                arrowprops={'arrowstyle': '->', 'lw': 1.5})
    ax.annotate('',
                (neb_distances[-1], barrier_energy),
                (neb_distances[-1], neb_enegies[-1]),
                arrowprops={'arrowstyle': '->', 'lw': 1.5})
    ax.plot([neb_distances[0], neb_distances[-1]],
            [barrier_energy, barrier_energy], 'k:')
    '''
    return ax


def get_barrier(images):
    """Get the barrier."""
    neb_distances = np.array([img.info['neb_distance'] for img in images])
    neb_enegies = np.array([img.get_potential_energy(FREE_ENERGY)
                            for img in images])
    neb_forces = np.array([img.info['neb_force'] for img in images])

    neb_enegies = neb_enegies

    neb_poly = CubicHermiteSpline(neb_distances, neb_enegies, -neb_forces)

    distance = np.linspace(neb_distances[0], neb_distances[-1], num=500)
    energy = neb_poly(distance)

    # Find the maximum of the neb
    barrier_index = np.argmax(energy)
    barrier_distance = distance[barrier_index]
    barrier_energy = energy[barrier_index]

    return (barrier_energy - neb_enegies[0],
            barrier_energy - neb_enegies[-1],
            neb_enegies[-1] - neb_enegies[0])


if __name__ == '__main__':
    import ase.io
    import sys
    from ase.visualize import view

    fig, ax = plt.subplots()

    images = []

    for fname in sys.argv[1:]:
        image = ase.io.read(fname)

        if isinstance(image, list):
            images.extend(image)
        else:
            images.append(image)

    set_neb_info(images, set_velocities=True)
    print_neb(images)
    plot_neb(images, ax)
    barrier = get_barrier(images)

    print(f'Forward barrier: {barrier[0]:16.6f}')
    print(f'Reverse barrier: {barrier[1]:16.6f}')
    print(f'Reaction energy: {barrier[2]:16.6f}')

    view(images)
    plt.show()
