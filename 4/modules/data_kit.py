from math import radians, cos, sin, asin, sqrt
import folium as folium

def haversine(coordinates1, coordinates2):
    """Takes two coordinates and uses the Haversine equation to calculate the distance"""
    lon1 = coordinates1[1]
    lat1 = coordinates1[0]
    lon2 = coordinates2[1]
    lat2 = coordinates2[0]
    
    # Change to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Apply the Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 3956
    return c * r
  
def get_centered_nyc_map():
  return folium.Map(location=[40.7128, -74.006], zoom_start=12)

def add_distance_to_poi_col(dataset, point_of_interest_coords, title, lat_header='Lat', lon_header='Lon'):
  """
  Takes a DataSet containing the cols 'Lat' and 'Lon'.
  Calculates the distance for each rows Lat and Lon with a given (Lat,Lon) Tuple
  Returns the DataSet with an additional column representing the distance to the Tuple
  """
  dataset[title] = dataset[[lat_header, lon_header]].apply(lambda x: haversine(point_of_interest_coords, tuple(x)), axis=1)
  return dataset