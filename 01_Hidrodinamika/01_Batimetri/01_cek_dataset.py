import xarray as xr

ds = xr.open_dataset(r"D:\ANCOL\Model\01_Hidrodinamika\01_Batimetri\L0master_v08-ref05_net.nc")
print(ds)
print(ds.variables)

#import pandas as pd

#xyz = pd.read_csv(
#    r"D:\ANCOL\Model\01_Hidrodinamika\01_Batimetri\EDIT_Batimetri-minus_Ancol_UTM_16MAR26.xyz",
#    sep=r"\s+",
#    header=None,
#    names=["x", "y", "z"]
#)

#print(xyz.head())
#print(xyz.describe())

