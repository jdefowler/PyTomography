from __future__ import annotations
from collections.abc import Sequence
import pytomography
import torch
import numpy as np
from pytomography.utils import get_1d_gaussian_kernel
from .shared import get_scanner_LUT, get_axial_trans_ids_from_info, get_detector_ids_from_trans_axial_ids

def compute_sinogram_ids(info: dict):
    """Computes the sinogram detector IDs in the correct order and shape.
    Args:
        None
    
    Returns:
        torch.Tensor: Sinogram IDs in the shape [N_crystals_per_ring, N_crystals_per_ring, 2, 2]
    """
    scanner_lut = get_scanner_LUT(info)
    nr_sectors_trans, nr_sectors_axial, nr_modules_axial, nr_modules_trans, nr_crystals_trans, nr_crystals_axial = info['rsectorTransNr'], info['rsectorAxialNr'], info['moduleAxialNr'], info['moduleTransNr'], info['crystalTransNr'], info['crystalAxialNr']
    nr_rings = nr_sectors_axial * nr_modules_axial * nr_crystals_axial
    nr_crystals_per_ring = nr_sectors_trans * nr_modules_trans * nr_crystals_trans
    min_sector_difference = info['min_rsector_difference']
    min_crystal_difference = min_sector_difference * nr_modules_trans * nr_crystals_trans
    angular_size = int(nr_crystals_per_ring / 2)
    radial_size = int(nr_crystals_per_ring - 2 * min_crystal_difference - 1)
    distance_crystal_id_0_to_first_sector_center = (nr_modules_trans * nr_crystals_trans) / 2

    angular = torch.arange(angular_size).unsqueeze(1)  # shape (A, 1)
    radial = torch.arange(radial_size).unsqueeze(0)   # shape (1, R)

    # Broadcast to shape (A, R)
    angular = angular.expand(angular_size, radial_size)
    radial = radial.expand(angular_size, radial_size)

    # Compute crystal indices
    id_a = (3 * nr_crystals_per_ring / 4) + (min_crystal_difference / 2) + torch.floor(radial / 2) + distance_crystal_id_0_to_first_sector_center - angular
    id_b = (3 * nr_crystals_per_ring / 4) - (min_crystal_difference / 2) - torch.floor((radial + 1) / 2) + distance_crystal_id_0_to_first_sector_center - angular

    # Wrap around with modulo
    id_a = id_a.long() % nr_crystals_per_ring
    id_b = id_b.long() % nr_crystals_per_ring

    sinogram_detector_ids = torch.stack([id_a, id_b], dim=-1)

    ring1_list = []
    ring2_list = []
    upper_ring_idx = nr_rings -1
    lower_ring_idx = 1
    while lower_ring_idx < nr_rings:
        for ring1 in range(lower_ring_idx, nr_rings+1):
            for _ in range(2 - int(ring1 == lower_ring_idx)):
                ring1_list.append(ring1)
        for ring1 in range(1, upper_ring_idx):
            for _ in range(2 - int(ring1 == upper_ring_idx-1)):
                ring1_list.append(ring1)
        for ring2 in range(1, upper_ring_idx + 2):
            for _ in range(2 - int(ring2 == upper_ring_idx+1)):
                ring2_list.append(ring2)
        for ring2 in range(lower_ring_idx + 2, nr_rings+1):
            for _ in range(2 - int(ring2 == lower_ring_idx+2)):
                ring2_list.append(ring2)
        lower_ring_idx += 2
        upper_ring_idx -= 2
    ring1_list.append(nr_rings)
    ring2_list.append(1)
    ring1_list = torch.tensor(ring1_list, dtype=torch.long)
    ring2_list = torch.tensor(ring2_list, dtype=torch.long)

    sinogram_ring_ids = torch.stack([ring1_list, ring2_list], dim=-1)
    
    return sinogram_detector_ids, sinogram_ring_ids

def sinogram_to_spatial(info: dict) -> Sequence[torch.Tensor]:
    """Returns two tensors: the first yields the detector coordinates (x1/y1/x2/y2) of each of the two crystals given the element of the sinogram (shape [N_crystals_per_ring, N_crystals_per_ring, 2, 2]), the second yields the ring coordinates (z1/z2) given two ring IDs (shape [Nrings*Nrings, 2])

    Args:
        info (dict): PET geometry information dictionary

    Returns:
        Sequence[torch.Tensor]: Two tensors yielding spatial coordinates
    """
    scanner_lut = get_scanner_LUT(info)
    nr_sectors_trans, nr_sectors_axial, nr_modules_axial, nr_modules_trans, nr_crystals_trans, nr_crystals_axial = info['rsectorTransNr'], info['rsectorAxialNr'], info['moduleAxialNr'], info['moduleTransNr'], info['crystalTransNr'], info['crystalAxialNr']
    nr_rings = nr_sectors_axial * nr_modules_axial * nr_crystals_axial
    nr_crystals_per_ring = nr_sectors_trans * nr_modules_trans * nr_crystals_trans
    min_sector_difference = info['min_rsector_difference']
    min_crystal_difference = min_sector_difference * nr_modules_trans * nr_crystals_trans
    angular_size = int(nr_crystals_per_ring / 2)
    radial_size = int(nr_crystals_per_ring - 2 * min_crystal_difference - 1)
    detector_coordinates = np.zeros((angular_size, radial_size, 2, 2), dtype=np.float32)
    ring_coordinates = np.zeros((nr_rings * nr_rings - nr_rings+1, 2), dtype=np.float32)

    sinogram_detector_ids, ring_ids = compute_sinogram_ids(info)

    detector_coordinates[:, :, 0, :] = scanner_lut[sinogram_detector_ids[:, :, 0], 0:2]
    detector_coordinates[:, :, 1, :] = scanner_lut[sinogram_detector_ids[:, :, 1], 0:2]

    ring_coordinates[:, 0] = scanner_lut[info['NrCrystalsPerRing']*(ring_ids[:,0]-1), 2]
    ring_coordinates[:, 1] = scanner_lut[info['NrCrystalsPerRing']*(ring_ids[:,1]-1), 2]

    return torch.tensor(detector_coordinates).to(torch.float32), torch.tensor(ring_coordinates).to(torch.float32)
   
def crystal_efficienies_to_sinogram(
    crystal_efficiencies: torch.Tensor,
    info: dict
) -> torch.Tensor:
    """Converts crystal efficiencies to a sinogram

    Args:
        crystal_efficiencies (torch.Tensor): Crystal efficiencies (shape [N_crystals_per_ring], [n_rings])
        info (dict): PET geometry information dictionary

    Returns:
        torch.Tensor: Sinogram of crystal efficiencies
    """
    scanner_lut = get_scanner_LUT(info)
    nr_sectors_trans, nr_sectors_axial, nr_modules_axial, nr_modules_trans, nr_crystals_trans, nr_crystals_axial = info['rsectorTransNr'], info['rsectorAxialNr'], info['moduleAxialNr'], info['moduleTransNr'], info['crystalTransNr'], info['crystalAxialNr']
    nr_rings = nr_sectors_axial * nr_modules_axial * nr_crystals_axial
    nr_crystals_per_ring = nr_sectors_trans * nr_modules_trans * nr_crystals_trans
    min_sector_difference = info['min_rsector_difference']
    min_crystal_difference = min_sector_difference * nr_modules_trans * nr_crystals_trans
    radial_size = int(nr_crystals_per_ring - 2 * min_crystal_difference - 1)
    angular_size = int(nr_crystals_per_ring / 2)
    distance_crystal_id_0_to_first_sector_center = (nr_modules_trans * nr_crystals_trans) / 2
    detector_coordinates = np.zeros((angular_size, radial_size, 2, 2), dtype=np.float32)
    ring_difference_size = nr_rings * nr_rings - nr_rings + 1
    crystal_sino = np.zeros((angular_size,radial_size,ring_difference_size))
    
    sinogram_detector_ids, ring_ids = compute_sinogram_ids(info)

    id_a = sinogram_detector_ids[:, :, 0].unsqueeze(-1).expand(-1, -1, ring_ids.shape[0])
    id_b = sinogram_detector_ids[:, :, 1].unsqueeze(-1).expand(-1, -1, ring_ids.shape[0])

    ring_id_a = ring_ids[:, 0].view(1, 1, -1).expand(sinogram_detector_ids.shape[0], sinogram_detector_ids.shape[1], -1)
    ring_id_b = ring_ids[:, 1].view(1, 1, -1).expand(sinogram_detector_ids.shape[0], sinogram_detector_ids.shape[1], -1)

    crystal_sino[:,:,:] = crystal_efficiencies[id_a,ring_id_a-1] * crystal_efficiencies[id_b,ring_id_b-1]

    return torch.tensor(crystal_sino)

def singles_to_randoms_sinogram(
    singles_rate: torch.Tensor,
    coincidence_window: np.float,
    info: dict
) -> torch.Tensor:
    """Converts singles to a randoms sinogram
    Args:
        singles (torch.Tensor): Singles data (shape [N_crystals_per_ring, N_crystals_per_ring, N_rings])
        coincidence_window (np.float): Coincidence window in seconds
        info (dict): PET geometry information dictionary
    Returns:
        torch.Tensor: Sinogram of randoms (shape [N_crystals_per_ring, N_crystals_per_ring, N_rings])
    """
    scanner_lut = get_scanner_LUT(info)
    nr_sectors_trans, nr_sectors_axial, nr_modules_axial, nr_modules_trans, nr_crystals_trans, nr_crystals_axial = info['rsectorTransNr'], info['rsectorAxialNr'], info['moduleAxialNr'], info['moduleTransNr'], info['crystalTransNr'], info['crystalAxialNr']
    nr_rings = nr_sectors_axial * nr_modules_axial * nr_crystals_axial
    nr_crystals_per_ring = nr_sectors_trans * nr_modules_trans * nr_crystals_trans
    min_sector_difference = info['min_rsector_difference']
    min_crystal_difference = min_sector_difference * nr_modules_trans * nr_crystals_trans
    radial_size = int(nr_crystals_per_ring - 2 * min_crystal_difference - 1)
    angular_size = int(nr_crystals_per_ring / 2)
    distance_crystal_id_0_to_first_sector_center = (nr_modules_trans * nr_crystals_trans) / 2
    detector_coordinates = np.zeros((angular_size, radial_size, 2, 2), dtype=np.float32)
    ring_difference_size = nr_rings * nr_rings - nr_rings + 1
    randoms_rate_sino = np.zeros((angular_size,radial_size,ring_difference_size))
    
    sinogram_detector_ids, ring_ids = compute_sinogram_ids(info)

    id_a = sinogram_detector_ids[:, :, 0].unsqueeze(-1).expand(-1, -1, ring_ids.shape[0])
    id_b = sinogram_detector_ids[:, :, 1].unsqueeze(-1).expand(-1, -1, ring_ids.shape[0])

    ring_id_a = ring_ids[:, 0].view(1, 1, -1).expand(sinogram_detector_ids.shape[0], sinogram_detector_ids.shape[1], -1)
    ring_id_b = ring_ids[:, 1].view(1, 1, -1).expand(sinogram_detector_ids.shape[0], sinogram_detector_ids.shape[1], -1)

    randoms_rate_sino[:,:,:] = 2 * coincidence_window * singles_rate[id_a,ring_id_a-1] * singles_rate[id_b,ring_id_b-1]

    return torch.tensor(randoms_rate_sino)