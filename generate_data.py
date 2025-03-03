import numpy as np
import pandas as pd
import pygrib

def compute_specific_humidity(temperature, relative_humidity, pressure):
    """
    Converts relative humidity (%) to specific humidity (kg/kg)
    using GFS variables: Temperature (K), Relative Humidity (%), Pressure (Pa).
    """

    saturation_vapor = np.zeros_like(temperature)

    # Convert temperature from Kelvin to Celsius
    T_C = temperature - 273.15  # Convert K to °C

    saturation_vapor[T_C > 0] = 0.61078 * np.exp((17.27 * T_C[T_C > 0]) / (T_C[T_C > 0] + 237.3))
    saturation_vapor[T_C <= 0] = 0.61078 * np.exp((21.875 * T_C[T_C <= 0]) / (T_C[T_C <= 0] + 265.5))

    #convert from kPa to Pa
    saturation_vapor = saturation_vapor * 1000

    ee = (relative_humidity / 100) * saturation_vapor

    #pressure in Pa
    ratio_molar = 0.622
    q = (ratio_molar * ee) / (pressure - (1 - ratio_molar) * ee) #kg vapor / kg air


    return q

def extract_values_from_grib_file3(file_path, parameter_name):
    # Open the GRIB file
    grbs = pygrib.open(file_path)

    # Get all the messages in the GRIB file
    messages = grbs.select()
    data = ''
    all_data = []
    all_level = []
    lats = ''
    lons = ''
    for grb in messages:
        level = grb.level
        print(parameter_name, level)
        
        #grb = grbs.select(parameterName=parameter_name, level=level)[0]
        grb = grbs.select(parameterName=parameter_name)[0]
        data = grb.values
        lats, lons = grb.latlons()
        all_data.append(data)
        all_level.append(level)

    all_data = np.array(all_data)
    all_level = np.array(all_level)
    if type(data) != type(''):
        grbs.close()
        return all_data, all_level, lats, lons
    else:
        raise Exception("data doesn't exist")

def extract_values_from_grib(file_path, parameter_name, level=None):
    # Open the GRIB file
    grbs = pygrib.open(file_path)

    if level is None:
        grb = grbs.select(parameterName=parameter_name)[0]
    else:
        grb = grbs.select(parameterName=parameter_name, level=level)[0]
    
    #keys = grb.keys()
    #for iter_k in range(len(keys)):
    #    if keys[iter_k] == 'values':
    #        continue
    #    print(keys[iter_k], '==', grb[keys[iter_k]])
    #exit()
    data = grb.values
    lats, lons = grb.latlons()

    return data, lats, lons



def list_grib_parameters(file_path):
    # Open the GRIB file
    grbs = pygrib.open(file_path)
    print('cek grbs', grbs)

    # Get all the messages in the GRIB file
    messages = grbs.select()

    # Loop through each message and print the parameter details
    param_name = []
    levels = []
    for grb in messages:
        parameter_name = grb.parameterName
        level = grb.level
        forecast_time = grb.forecastTime
        # Add more attributes as needed
        #print(parameter_name)
        cek = grbs.select(parameterName=parameter_name, level=level)[0]
        param_name.append(parameter_name)
        levels.append(level)
        keys = grb.keys()
        #print(cek, level, parameter_name)
        #print(keys)
        #break
        #print("Parameter Name: %s, level: %s, forecast time: %s"%(parameter_name, level, forecast_time))
        # Print more attributes as needed

    # Close the GRIB file
    grbs.close()
    return param_name, levels

precips_acc = []

precips = []
RHs = []
Qs = []
Temps = []
Clouds = []
Us = []
Vs = []
U10s = []
V10s = []
Cloudsurfaces = []

#maxs = []
#mins = []
#level_used = [100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 850, 900, 950, 1000]
#level_used = [700, 800, 850, 900, 950, 1000]
#level_used = [850, 900, 950, 1000]
level_used = [850, 1000]
for iter_d in range(1, 49):
    single_file = f'dataset/gfs.t12z.pgrb2.0p25.f{iter_d:03d}'
    #print(single_file)
    #list_grib_parameters(single_file)
    
    precip, lat, lon = extract_values_from_grib(single_file, 'Total precipitation')
    #maxs.append(np.nanmax(precip))
    precips_acc.append(precip)

    if iter_d == 1:
        precips.append(precip)
    elif ((iter_d-1) % 6) == 0:
        precips.append(precip)
    else:
        precip = precip - precips_acc[-2]
        precips.append(precip)
    
    Rhs_temp = []
    Temps_temp = []
    Clouds_temp = []
    Us_temp = []
    Vs_temp = []
    Qs_temp = []
    U10s_temp = []
    V10s_temp = []
    Cloudsurfaces_temp = []
    for iter_l in range(len(level_used)):
        RH, lat, lon = extract_values_from_grib(single_file, 'Relative humidity', level=level_used[iter_l])
        Rhs_temp.append(RH)

        #save lat lon
        if iter_d == 1 and iter_l == 0:
            np.save('lat.npy', lat)
            np.save('lon.npy', lon)


        Temp, lat, lon = extract_values_from_grib(single_file, 'Temperature', level=level_used[iter_l])
        Temps_temp.append(Temp)
        Cloud, lat, lon = extract_values_from_grib(single_file, 'Total cloud cover', level=level_used[iter_l])
        Clouds_temp.append(Cloud)
        U, lat, lon = extract_values_from_grib(single_file, 'u-component of wind', level=level_used[iter_l])
        Us_temp.append(U)
        V, lat, lon = extract_values_from_grib(single_file, 'v-component of wind', level=level_used[iter_l])
        Vs_temp.append(V)

        q = compute_specific_humidity(Temp, RH, level_used[iter_l]*100)
        Qs_temp.append(q)
    
    U10, lat, lon = extract_values_from_grib(single_file, 'u-component of wind', level=10)
    V10, lat, lon = extract_values_from_grib(single_file, 'v-component of wind', level=10)
    Cloudsurface, lat, lon = extract_values_from_grib(single_file, 'Total cloud cover', level=0)

    Rhs_temp = np.array(Rhs_temp)
    Temps_temp = np.array(Temps_temp)
    Clouds_temp = np.array(Clouds_temp)
    Us_temp = np.array(Us_temp)
    Vs_temp = np.array(Vs_temp)
    Qs_temp = np.array(Qs_temp)

    RHs.append(Rhs_temp)
    Temps.append(Temps_temp)
    Clouds.append(Clouds_temp)
    Us.append(Us_temp)
    Vs.append(Vs_temp)
    Qs.append(Qs_temp)
    
    U10s.append(U10)
    V10s.append(V10)
    Cloudsurfaces.append(Cloudsurface)



precips = np.array(precips)
for iter_p in range(len(precips)):
    print(np.nanmin(precips[iter_p]), np.nanmax(precips[iter_p]), iter_p)
#maxs = np.array(maxs)

Rhs = np.array(RHs)
Temps = np.array(Temps)
Clouds = np.array(Clouds)
Us = np.array(Us)
Vs = np.array(Vs)
Qs = np.array(Qs)
U10s = np.array(U10s)
V10s = np.array(V10s)
Cloudsurfaces = np.array(Cloudsurfaces)

np.save('precips.npy', precips)
np.save('Rhs.npy', Rhs)
np.save('Temps.npy', Temps)
np.save('Clouds.npy', Clouds)
np.save('Us.npy', Us)
np.save('Vs.npy', Vs)
np.save('Qs.npy', Qs)
np.save('U10s.npy', U10s)
np.save('V10s.npy', V10s)
np.save('Cloudsurfaces.npy', Cloudsurfaces)