import cupy as np


def do_heat_transfer(matrix_a, matrix_b, mask,
                     heat_capacity_coefficient_a=100.0,
                     heat_capacity_coefficient_b=1.0,
                     heat_transfer_coefficient=0.1):
    temperature_delta_b = (matrix_a - matrix_b) * heat_transfer_coefficient * mask
    energy_delta = temperature_delta_b * heat_capacity_coefficient_b
    temperature_delta_a = -energy_delta / heat_capacity_coefficient_a
    return matrix_a + temperature_delta_a, matrix_b + temperature_delta_b


def build_irradiance_mask(water, water_coef, soil_coef):
    irradiance_mask = (water / water_coef.heat_capacity * (1 - water_coef.albedo)
                       + (1 - water) / soil_coef.heat_capacity * (1 - water_coef.albedo))
    return irradiance_mask


def map_matrix_padding(map_matrix, padding=10):
    l_padding = np.copy(map_matrix[:, :padding])
    r_padding = np.copy(map_matrix[:, -padding:])
    map_matrix = np.concatenate((r_padding, map_matrix, l_padding), axis=1)

    d_padding = np.flip(np.copy(map_matrix[:padding]), axis=0)
    u_padding = np.flip(np.copy(map_matrix[-padding:]), axis=0)
    map_matrix = np.concatenate((d_padding, map_matrix, u_padding), axis=0)
    return map_matrix


def map_matrix_repadding(map_matrix, padding=10):
    return map_matrix[padding:-padding, padding:-padding]
