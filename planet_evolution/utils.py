from jax import jit

@jit
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
