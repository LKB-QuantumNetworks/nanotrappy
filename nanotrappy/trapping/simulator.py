import itertools
import numpy as np
from scipy import linalg as LA
from tqdm.contrib.concurrent import process_map
from nanotrappy.utils.physicalunits import *
from nanotrappy.utils.utils import *


class Simulator:
    @staticmethod
    def convert_fields_to_spherical_basis(Ex, Ey, Ez):  ## returns Ep, Em, E0 (quantization axis 'z')
        return -(Ex + Ey * 1j) / np.sqrt(2), (Ex - Ey * 1j) / np.sqrt(2), Ez

    def parallel_beam_potential(self, *args):
        i, j, atomicsystem, Etot = args[0]
        E0, Ep, Em = self.convert_fields_to_spherical_basis(Etot[0, i, j], Etot[1, i, j], Etot[2, i, j])
        val, vec = atomicsystem.potential(Ep, Em, E0)
        idx = val.argsort()
        return val[idx], vec[idx]

    def parallel_beam_pair_potential(self, *args):
        i, j, atomicsystem, alpha0_f, alpha1_f, alpha2_f, alpha0_b, alpha1_b, alpha2_b, E_fwd, E_bwd = args[0]
        Ep, Em, E0 = self.convert_fields_to_spherical_basis(E_fwd[0, i, j], E_fwd[1, i, j], E_fwd[2, i, j])
        tot_shift_f = (
            -1.0 * alpha0_f * atomicsystem.delta_scalar(Ep, Em, E0)
            - 1.0 * alpha1_f * atomicsystem.delta_vector(Ep, Em, E0)
            - 1.0 * alpha2_f * atomicsystem.delta_tensor(Ep, Em, E0)
        )
        Ep, Em, E0 = self.convert_fields_to_spherical_basis(E_bwd[0, i, j], E_bwd[1, i, j], E_bwd[2, i, j])
        tot_shift_b = (
            -1.0 * alpha0_b * atomicsystem.delta_scalar(Ep, Em, E0)
            - 1.0 * alpha1_b * atomicsystem.delta_vector(Ep, Em, E0)
            - 1.0 * alpha2_b * atomicsystem.delta_tensor(Ep, Em, E0)
        )
        tot_shift = (tot_shift_f + tot_shift_b) / (mK * kB)
        val, vec = LA.eig(tot_shift)
        idx = val.argsort()
        return val[idx], vec[idx]


class SequentialSimulator(Simulator):
    """ """

    def simulate(self, simulation, potential_number, mf_shift):
        size = simulation.Etot.shape[1:]

        ranges = [range(s) for s in size]
        # ranges[0] = progressbar(ranges[0], "\n Computing: ", 40)
        for idx in itertools.product(*ranges):
            Ep, Em, E0 = self.convert_fields_to_spherical_basis(
                simulation.Etot[(0, *idx)], simulation.Etot[(1, *idx)], simulation.Etot[(2, *idx)]
            )
            val, vec = simulation.atomicsystem.potential(Ep, Em, E0)
            simulation.potentials[(potential_number, *idx)] = np.real(val[mf_shift])
            for id, elt in enumerate(vec):
                simulation.vecs[(potential_number, *idx, id)] = elt

    def simulate_pair(self, simulation, beam, E_fwd, E_bwd, potential_number, mf_shift):
        size = simulation.Etot.shape[1:]

        alpha0_f, alpha1_f, alpha2_f = simulation.atomicsystem.set_alphas(beam.get_lmbda()[0])
        alpha0_b, alpha1_b, alpha2_b = simulation.atomicsystem.set_alphas(beam.get_lmbda()[1])

        ranges = [range(s) for s in size]
        # ranges[0] = progressbar(ranges[0], "\n Computing: ", 40)
        for idx in itertools.product(*ranges):
            Ep, Em, E0 = self.convert_fields_to_spherical_basis(E_fwd[(0, *idx)], E_fwd[(1, *idx)], E_fwd[(2, *idx)])
            tot_shift_f = (
                -1.0 * alpha0_f * simulation.atomicsystem.delta_scalar(Ep, Em, E0)
                - 1.0 * alpha1_f * simulation.atomicsystem.delta_vector(Ep, Em, E0)
                - 1.0 * alpha2_f * simulation.atomicsystem.delta_tensor(Ep, Em, E0)
            )

            Ep, Em, E0 = self.convert_fields_to_spherical_basis(E_bwd[(0, *idx)], E_bwd[(1, *idx)], E_bwd[(2, *idx)])
            tot_shift_b = (
                -1.0 * alpha0_b * simulation.atomicsystem.delta_scalar(Ep, Em, E0)
                - 1.0 * alpha1_b * simulation.atomicsystem.delta_vector(Ep, Em, E0)
                - 1.0 * alpha2_b * simulation.atomicsystem.delta_tensor(Ep, Em, E0)
            )

            tot_shift = (tot_shift_f + tot_shift_b) / (mK * kB)
            vals, vec = LA.eig(tot_shift)
            idx_order = vals.argsort()

            simulation.potentials[(potential_number, *idx)][:] = np.real(vals[idx_order][mf_shift])
            for id, elt in enumerate(vec):
                simulation.vecs[(potential_number, *idx, id)][:] = elt


class ParallelSimulator(Simulator):
    """[summary]

    Args:
        Simulator ([type]): [description]
    """

    def __init__(self, max_workers=4):
        self.max_workers = max_workers

    def simulate(self, simulation, potential_number, mf_shift):

        if simulation.geometry.dimension == 1:
            seq_sim = SequentialSimulator()
            seq_sim.simulate(simulation, potential_number, mf_shift)
        else:
            dim1, dim2 = simulation.Etot.shape[1:]
            size = simulation.Etot.shape[1:]
            input = (
                (i, j, simulation.atomicsystem, simulation.Etot)
                for i, j in itertools.product(range(dim2), repeat=2)
                if i < dim1
            )

            output = ((i, j) for i, j in itertools.product(range(dim2), repeat=2) if i < dim1)

            results = process_map(
                self.parallel_beam_potential,
                list(input),
                max_workers=self.max_workers,
                chunksize=int(dim1 * dim2 / 100.0),
            )

            for i, elt in enumerate(output):
                simulation.potentials[potential_number][elt][:] = np.real(results[i][0][mf_shift])
                for id, elt_vec in enumerate(results[i][1]):
                    simulation.vecs[potential_number][elt][id][:] = elt_vec

    def simulate_pair(self, simulation, beam, E_fwd, E_bwd, potential_number, mf_shift):
        if simulation.geometry.dimension == 1:
            seq_sim = SequentialSimulator()
            seq_sim.simulate_pair(simulation, beam, E_fwd, E_bwd, potential_number, mf_shift)
        else:
            alpha0_f, alpha1_f, alpha2_f = simulation.atomicsystem.set_alphas(beam.get_lmbda()[0])
            alpha0_b, alpha1_b, alpha2_b = simulation.atomicsystem.set_alphas(beam.get_lmbda()[1])

            # dim1 = len(simulation.coord1)
            # dim2 = len(simulation.coord2)
            dim1, dim2 = simulation.Etot.shape[1:]

            input = (
                (
                    i,
                    j,
                    simulation.atomicsystem,
                    alpha0_f,
                    alpha1_f,
                    alpha2_f,
                    alpha0_b,
                    alpha1_b,
                    alpha2_b,
                    E_fwd,
                    E_bwd,
                )
                for i, j in itertools.product(range(dim2), repeat=2)
                if i < dim1
            )
            output = ((i, j) for i, j in itertools.product(range(dim2), repeat=2) if i < dim1)

            results = process_map(
                self.parallel_beam_pair_potential,
                list(input),
                max_workers=self.max_workers,
                chunksize=int(dim1 * dim2 / 100.0),
            )
            for i, elt in enumerate(output):
                simulation.potentials[potential_number][elt][:] = 0.5 * np.real(results[i][0][mf_shift])
                for id, elt_vec in enumerate(results[i][1]):
                    simulation.vecs[potential_number][elt][id][:] = elt_vec
