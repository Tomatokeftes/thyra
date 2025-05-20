from napari.utils import Colormap
from spatialdata_plot.pl.utils import set_zero_in_cmap_to_transparent

from napari_spatialdata import Interactive
from spatialdata import SpatialData as sd
from spatialdata import rasterize_bins


sdata=sd.read(r"C:\Users\P70078823\Desktop\MSIConverter\pea8.zarr")
sdata["msi_dataset"].X = sdata["msi_dataset"].X.tocsc()
rasterized = rasterize_bins(
    sdata,
    'msi_dataset_pixels',
    "msi_dataset",
    'x',
    'y',
)
new_cmap = set_zero_in_cmap_to_transparent(cmap="viridis")
napari_cmap = Colormap(new_cmap.colors, "viridis_zero_transparent")
sdata['rasterized'] = rasterized
interactive = Interactive(sdata, headless=True)
interactive.add_element(f"rasterized", "global", view_element_system=True)
interactive.run()
