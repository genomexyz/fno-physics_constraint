import numpy as np

def calculate_saturation_vapor(temperature):
    saturation_vapor = np.zeros_like(temperature)

    # Convert temperature from Kelvin to Celsius
    T_C = temperature - 273.15  # Convert K to °C

    saturation_vapor[T_C > 0] = 0.61078 * np.exp((17.27 * T_C[T_C > 0]) / (T_C[T_C > 0] + 237.3))
    saturation_vapor[T_C <= 0] = 0.61078 * np.exp((21.875 * T_C[T_C <= 0]) / (T_C[T_C <= 0] + 265.5))

    #convert from kPa to Pa
    saturation_vapor = saturation_vapor * 1000
    return saturation_vapor

def calculate_slope_saturation_vapor(temperature):
    es = calculate_saturation_vapor(temperature)
     # Convert temperature from Kelvin to Celsius
    T_C = temperature - 273.15  # Convert K to °C
    ses = (4098 * es) / ((T_C + 237.3) ** 2)
    return ses

def calculate_gamma(pressure):
    Cp = 1005  # Specific heat of dry air (J/kg·K)
    Lambda = 2.45e6  # Latent heat of vaporization (J/kg)
    R = 287  # Gas constant for dry air (J/kg·K)
    gamma = (Cp * pressure) / (Lambda * R)  # Psychrometric constant
    return gamma

def calculate_gamma2(pressure):
    Cp = 1.0035
    Lambda = 2.45
    MW = 0.622
    gamma = (Cp * pressure) / (Lambda * MW)
    return gamma

def calculate_Rs(cloud_cover, pressure):
    S0 = 1367  # Solar constant (W/m^2)
    cloud_cover_fraction = cloud_cover / 100
    Rs = S0 * (0.75 + (2e-5 * pressure)) * (1 - (0.75 * cloud_cover_fraction))
    return Rs

def calculate_hourly_extraterrestrial_radiation(lat, day_of_year, hour):
    """
    Computes hourly extraterrestrial solar radiation (R_a_hour) in MJ/m²/h
    using latitude, day of the year, and hour of the day.

    Parameters:
    - lat: Latitude in degrees
    - day_of_year: Day of the year (1-365)
    - hour: Hour of the day (0-23)
    
    Returns:
    - R_a_hour (MJ/m²/h)
    """

    # Constants
    G_sc = 0.0820  # Solar constant (MJ/m²/min)
    
    # Convert latitude to radians
    lat_rad = np.radians(lat)

    # Declination angle (δ) in radians
    declination = 0.409 * np.sin(2 * np.pi * day_of_year / 365 - 1.39)

    # Inverse relative distance Earth-Sun (dr)
    dr = 1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365)

    # Compute the hour angle for the start and end of the hour
    omega1 = np.pi * (hour - 0.5) / 12 - np.pi  # Start of the hour
    omega2 = np.pi * (hour + 0.5) / 12 - np.pi  # End of the hour

    # Extraterrestrial radiation for the given hour
    R_a_hour = (12 * 60 / np.pi) * G_sc * dr * (
        (np.sin(lat_rad) * np.sin(declination) * (omega2 - omega1)) +
        (np.cos(lat_rad) * np.cos(declination) * (np.sin(omega2) - np.sin(omega1)))
    )

    return max(R_a_hour, 0)  # Ensure non-negative values

#def calculate_Rs2(cloud_cover, latitude):
#    As = 0.25
#    Bs = 0.5
#    Rs = (As + (Bs * cloud_cover)) * Ra


def calculate_ET(delta, gamma, cloud_cover, pressure):
    Rs = calculate_Rs(cloud_cover, pressure)
    ET = 0.65 * (delta / (delta + gamma)) #* (Rs / gamma)
    #print('cek RS gamma', Rs, gamma)
    #print('cek delta', delta)
    #exit()
    return ET

def compute_spatial_derivatives(q, res):
    """
    Computes dq/dx and dq/dy using finite differences on a 2D specific humidity field.

    Parameters:
    - q (numpy.ndarray): 2D array of specific humidity (shape: [lat, lon])
    - lon (numpy.ndarray): 1D array of longitude values (degrees)
    - lat (numpy.ndarray): 1D array of latitude values (degrees)

    Returns:
    - dq_dx (numpy.ndarray): Gradient of q along the x-direction (longitude)
    - dq_dy (numpy.ndarray): Gradient of q along the y-direction (latitude)
    """

    # Convert latitude and longitude degrees to meters (approximate)
    #R = 6371000  # Earth's radius in meters
    #dlon = np.deg2rad(np.mean(np.diff(lon))) * R * np.cos(np.deg2rad(np.mean(lat)))  # Longitude spacing
    #dlat = np.deg2rad(np.mean(np.diff(lat))) * R  # Latitude spacing

    degree2meter = 111111
    dlon = res * degree2meter
    dlat = res * degree2meter

    # Compute dq/dx (longitude gradient) using central difference
    dq_dx = np.zeros_like(q)
    dq_dx[:, 1:-1] = (q[:, 2:] - q[:, :-2]) / (2 * dlon)
    dq_dx[:, 0] = dq_dx[:, 1]  # Handle left boundary
    dq_dx[:, -1] = dq_dx[:, -2]  # Handle right boundary

    # Compute dq/dy (latitude gradient) using central difference
    dq_dy = np.zeros_like(q)
    dq_dy[1:-1, :] = (q[2:, :] - q[:-2, :]) / (2 * dlat)
    dq_dy[0, :] = dq_dy[1, :]  # Handle bottom boundary
    dq_dy[-1, :] = dq_dy[-2, :]  # Handle top boundary

    return dq_dx, dq_dy

precips = np.load('precips.npy')
Rhs = np.load('Rhs.npy')
Temps = np.load('Temps.npy')
Clouds = np.load('Clouds.npy')
Us = np.load('Us.npy')
Vs = np.load('Vs.npy')
Qs = np.load('Qs.npy')
Cloudsurface = np.load('Cloudsurfaces.npy')
U10s = np.load('U10s.npy')
V10s = np.load('V10s.npy')

lat2d = np.load('lat.npy')
lat = lat2d[:, 0]
lon2d = np.load('lon.npy')
lon = lon2d[0, :]

#Rs = calculate_Rs(Clouds, 100000)
#gamma = calculate_gamma(100000)
#print(Rs)
#print('===================================')
#print(gamma)

#delta = calculate_slope_saturation_vapor(Temps)
#gamma = calculate_gamma(100000)
#ET = calculate_ET(delta, gamma, Clouds, 100000)
#print(ET)

dxy = 0.25
for iter_d in range(1, len(Temps)):
    precip = precips[iter_d, :, :]
    #U10 = U10s[iter_d, :, :]
    #V10 = V10s[iter_d, :, :]
    U1000 = Us[iter_d, -1, :, :]
    V1000 = Vs[iter_d, -1, :, :]

    U850 = Us[iter_d, 0, :, :]
    V850 = Vs[iter_d, 0, :, :]

    Temperature = Temps[iter_d, -1, :, :]
    Relative_humidity = Rhs[iter_d, -1, :, :]
    Pressure = 100000
    Cloud = Cloudsurface[iter_d, :, :]

    Q = Qs[iter_d, -1, :, :]
    Q_past = Qs[iter_d-1, -1, :, :]
    dq_dt = (Q - Q_past)
    dq_dx, dq_dy = compute_spatial_derivatives(Q, dxy)

    print('shape dq', dq_dx.shape, dq_dy.shape)
    print('shape Q', Q.shape)
 

    delta = calculate_slope_saturation_vapor(Temperature)
    gamma = calculate_gamma2(100)
    ET = calculate_ET(delta, gamma, Cloud, 100000)
    print(ET.shape, delta.shape, Cloud.shape)

    minET = precip + dq_dt + (U1000 * dq_dx) + (V1000 * dq_dy) + (U850 * dq_dx) + (V850 * dq_dy)
    loss_physics = ET - minET
    #print(loss_physics, loss_physics.shape)
    print('cek ET', np.nanmin(ET), np.nanmean(ET), np.nanmax(ET))
    print('cek minET', np.nanmin(minET), np.nanmean(minET), np.nanmax(minET))
    print('cek precip', np.nanmin(precip), np.nanmean(precip), np.nanmax(precip))
    print(ET, np.nanmin(ET), np.nanmean(ET), np.nanmax(ET))
    #break

