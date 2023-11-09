import h5py
import numpy as np
import torch
from einops import rearrange
from matplotlib import pyplot as plt


def display_sample_image(base_path, h5_file_path, FIPS, date, grid_num):
    """Function to return grid image based on a provided h5 file and parameters

    Parameters
    ----------
    base_path: (String) path to where the Sentinel data was downloaded
    h5_file_path: (String) path to the particular h5 file that has images to display
    FIPS: (String) the FIPS code of the county you want to represent
    date: (String) the date that you want images from (Example: "2021-04-01")
    grid_num: (integer) the grid number you want to display

    Returns
    ----------
    None
    """
    # read the h5 file
    h5_data = h5py.File(base_path + h5_file_path, 'r')

    # get the particular county information
    FIPS_data = h5_data[FIPS]

    # get the particular date information
    date_data = FIPS_data[date]

    # print the state of the grid
    state = date_data["state"][0].decode()
    state = date_data["state"]
    print("State of the Grid:", state)
    print()

    # print the county of the grid
    county = date_data["county"][0].decode()
    county = date_data["county"]
    print("County of the Grid:", county)
    print()

    # print the coordinates of the particular grid
    coordinates = date_data["coordinates"][grid_num]
    print("Coordinates of the grid:")
    print("lower left corner lat:", round(coordinates[0][0], 2))
    print("lower left corner lon:", round(coordinates[0][1], 2))
    print("upper right corner lat:", round(coordinates[1][0], 2))
    print("upper right corner lon:", round(coordinates[1][1], 2))

    # display the image of the chosen grid
    x = np.asarray(date_data["data"])
    x = rearrange(x, 'b h w c -> (b h) w c')
    print(x.shape)
    image_data = x

    # show the image
    plt.imshow(image_data)
    plt.show()


if __name__ == "__main__":
    base_path = r"C:\Users\MSI\Downloads"
    file_path = r"\Agriculture_19_IA_2017-01-01_2017-03-31.h5"
    display_sample_image(base_path, file_path, "19001", "2017-01-01", 0)
