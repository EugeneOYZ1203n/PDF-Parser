"""The shared P2 raster prelude every P2 backend calls first
(`raster_text.prepare_images`): tiled PaddleOCR on the raw image, colour
CLAHE (LAB L only), a simple colour-layer split (achromatic + hue-histogram
peaks), and erasing each recognized word's ink layer. Shared through `commons/`, like every
helper more than one backend uses."""
