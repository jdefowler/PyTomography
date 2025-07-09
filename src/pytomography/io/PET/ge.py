from __future__ import annotations
from collections.abc import Sequence
import pytomography
import torch
import numpy as np
from pytomography.utils import get_1d_gaussian_kernel
from .shared import get_axial_trans_ids_from_info, get_detector_ids_from_trans_axial_ids


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

    angular = torch.arange(angular_size).unsqueeze(1)  # shape (A, 1)
    radial = torch.arange(radial_size).unsqueeze(0)   # shape (1, R)

    # Broadcast to shape (A, R)
    angular = angular.expand(angular_size, radial_size)
    radial = radial.expand(angular_size, radial_size)

    # Compute crystal indices
    id_a = (3 * nr_crystals_per_ring / 4) + (min_crystal_difference / 2) + torch.floor(radial / 2) - angular
    id_b = (3 * nr_crystals_per_ring / 4) - (min_crystal_difference / 2) - torch.floor((radial + 1) / 2) - angular

    # Wrap around with modulo
    id_a = id_a.long() % nr_crystals_per_ring
    id_b = id_b.long() % nr_crystals_per_ring

    sinogram_detector_ids = torch.stack([id_a, id_b], dim=-1)

    ring_coordinates = np.zeros((nr_rings * nr_rings - nr_rings+1, 2), dtype=np.float32)
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

def get_scanner_LUT(info: dict):
    """Obtains scanner lookup table (gives x/y/z coordinates for each detector ID)

    Args:
        info (dict): PET geometry information dictionary

    Returns:
        torch.Tensor[N_detectors, 3]: Lookup table
    """
    ids_trans_crystal, ids_axial_crystal, ids_trans_submodule, ids_axial_submodule, ids_trans_module, ids_axial_module, ids_trans_rsector, ids_axial_rsector = get_axial_trans_ids_from_info(info)
    ids_detector = get_detector_ids_from_trans_axial_ids(ids_trans_crystal, ids_trans_submodule, ids_trans_module, ids_trans_rsector, ids_axial_crystal, ids_axial_submodule, ids_axial_module, ids_axial_rsector, info)
    Z_modules = ids_axial_module * info['moduleAxialSpacing'] - (info['moduleAxialNr']-1) * info['moduleAxialSpacing'] / 2
    Z_submodules = Z_modules + ids_axial_submodule * info['submoduleAxialSpacing'] - (info['submoduleAxialNr']-1) * info['submoduleAxialSpacing'] / 2
    Z_crystals = Z_submodules + ids_axial_crystal * info['crystalAxialSpacing'] - (info['crystalAxialNr']-1) * info['crystalAxialSpacing'] / 2
    # Get X/Y position of crystals
    # Start by getting X/Y position inside a submodule aligned with the center of the scanner
    Y_modules = ids_trans_module * info['moduleTransSpacing'] - (info['moduleTransNr']-1) * info['moduleTransSpacing'] / 2
    Y_submodules = Y_modules + ids_trans_submodule * info['submoduleTransSpacing'] - (info['submoduleTransNr']-1) * info['submoduleTransSpacing'] / 2
    Y_crystals = Y_submodules +ids_trans_crystal * info['crystalTransSpacing'] - (info['crystalTransNr']-1) * info['crystalTransSpacing'] / 2
    X_crystals = info['radius'] * torch.ones(len(Y_crystals))
    # Now apply rotation based on angle of the rsector
    angle_rsector = ids_trans_rsector / info['rsectorTransNr'] * 2 * np.pi
    rotation_matrices = torch.stack([
        torch.cos(angle_rsector),
        -torch.sin(angle_rsector),
        torch.sin(angle_rsector),
        torch.cos(angle_rsector)], dim=1).view(-1, 2, 2)
    XY_crystals = torch.vstack([X_crystals, Y_crystals])
    XY_crystals = torch.einsum('ijk,ki->ij', rotation_matrices, XY_crystals)
    if info['firstCrystalAxis'] == 0: # crystal with index 0 along x axis
        XY_crystals = XY_crystals.flip(dims=(1,))
    # Stack all together (for some reason Z needs to be negative?)
    XYZ_crystals = torch.vstack([XY_crystals.T, -Z_crystals.unsqueeze(0)]).T
    # Now sort scanner LUT by ids_detector order
    XYZ_crystals = XYZ_crystals[torch.argsort(ids_detector)]
    
    return XYZ_crystals
   
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

def get_detector_ids_from_trans_axial_ids(
    ids_trans_crystal: torch.Tensor,
    ids_trans_submodule: torch.Tensor,
    ids_trans_module: torch.Tensor,
    ids_trans_rsector: torch.Tensor,
    ids_axial_crystal: torch.Tensor,
    ids_axial_submodule: torch.Tensor,
    ids_axial_module: torch.Tensor,
    ids_axial_rsector: torch.Tensor,
    info: dict
    ) -> torch.Tensor:
    """Obtain detector IDs from individual part IDs

    Args:
        ids_trans_crystal (torch.Tensor): Transaxial crystal IDs
        ids_trans_submodule (torch.Tensor): Transaxial submodule IDs
        ids_trans_module (torch.Tensor): Transaxial module IDs
        ids_trans_rsector (torch.Tensor): Transaxial rsector IDs
        ids_axial_crystal (torch.Tensor): Axial crystal IDs
        ids_axial_submodule (torch.Tensor): Axial submodule IDs 
        ids_axial_module (torch.Tensor): Axial module IDs
        ids_axial_rsector (torch.Tensor): Axial rsector IDs 
        info (dict): PET geometry information dictionary

    Returns:
        torch.Tensor: Tensor containing (spatial) detector IDs
    """
    ids_ring = ids_axial_crystal +\
        ids_axial_submodule * info['crystalAxialNr'] +\
        ids_axial_module * info['crystalAxialNr'] * info['submoduleAxialNr'] +\
        ids_axial_rsector * info['crystalAxialNr'] * info['submoduleAxialNr'] * info['moduleAxialNr']
    ids_within_ring = ids_trans_crystal +\
        ids_trans_submodule * info['crystalTransNr'] +\
        ids_trans_module * info['crystalTransNr'] * info['submoduleTransNr'] +\
        ids_trans_rsector * info['crystalTransNr'] * info['submoduleTransNr'] * info['moduleTransNr']
    nb_crystal_per_ring = info['crystalTransNr'] * info['moduleTransNr'] * info['submoduleTransNr'] * info['rsectorTransNr']
    ids_detector = ids_ring * nb_crystal_per_ring + ids_within_ring
    return ids_detector

def get_axial_trans_ids_from_info(
    info: dict,
    return_combinations: bool = False,
    sort_by_detector_ids: bool = False
    ):
    """Get axial and transaxial IDs corresponding to each crystal in the scanner

    Args:
        info (dict): PET geometry information dictionary
        return_combinations (bool, optional): Whether or not to return all possible combinations (crystal pairs). Defaults to False.
        sort_by_detector_ids (bool, optional): Whether or not to sort by increasing detector IDs. Defaults to False.

    Returns:
        Sequence[torch.Tensor]: IDs corresponding to axial/transaxial components of each part
    """
    ids_trans_crystal = torch.arange(0, info['crystalTransNr'])
    ids_axial_crystal = torch.arange(0, info['crystalAxialNr'])
    ids_trans_submodule = torch.arange(0, info['submoduleTransNr'])
    ids_axial_submodule = torch.arange(0, info['submoduleAxialNr'])
    ids_trans_module = torch.arange(0, info['moduleTransNr'])
    ids_axial_module = torch.arange(0, info['moduleAxialNr'])
    ids_trans_rsector = torch.arange(0, info['rsectorTransNr'])
    ids_axial_rsector = torch.arange(0, info['rsectorAxialNr'])
    ids_trans_crystal, ids_axial_crystal, ids_trans_submodule, ids_axial_submodule, ids_trans_module, ids_axial_module, ids_trans_rsector, ids_axial_rsector = torch.cartesian_prod(ids_trans_crystal, ids_axial_crystal, ids_trans_submodule, ids_axial_submodule, ids_trans_module, ids_axial_module, ids_trans_rsector, ids_axial_rsector).T
    if sort_by_detector_ids:
        ids_detector = get_detector_ids_from_trans_axial_ids(ids_trans_crystal, ids_trans_submodule, ids_trans_module, ids_trans_rsector, ids_axial_crystal, ids_axial_submodule, ids_axial_module, ids_axial_rsector, info)
        idx_sort = torch.argsort(ids_detector)
        ids_trans_crystal = ids_trans_crystal[idx_sort]
        ids_axial_crystal = ids_axial_crystal[idx_sort]
        ids_trans_submodule = ids_trans_submodule[idx_sort]
        ids_axial_submodule = ids_axial_submodule[idx_sort]
        ids_trans_module = ids_trans_module[idx_sort]
        ids_axial_module = ids_axial_module[idx_sort]
        ids_trans_rsector = ids_trans_rsector[idx_sort]
        ids_axial_rsector = ids_axial_rsector[idx_sort]
    if return_combinations:
        ids_trans_crystal = torch.combinations(ids_trans_crystal, 2)
        ids_axial_crystal = torch.combinations(ids_axial_crystal, 2)
        ids_trans_submodule = torch.combinations(ids_trans_submodule, 2)
        ids_axial_submodule = torch.combinations(ids_axial_submodule, 2)
        ids_trans_module = torch.combinations(ids_trans_module, 2)
        ids_axial_module = torch.combinations(ids_axial_module, 2)
        ids_trans_rsector = torch.combinations(ids_trans_rsector, 2)
        ids_axial_rsector = torch.combinations(ids_axial_rsector, 2)
    return ids_trans_crystal, ids_axial_crystal, ids_trans_submodule, ids_axial_submodule, ids_trans_module, ids_axial_module, ids_trans_rsector, ids_axial_rsector