# DIQA (Document Image Quality Assessment)

diqa.py contains four different document image quality assessment metrics: (1) skewness, (2) contrast, (3) range-effect, and (4) bleed-through.

## Requirements

The following two packages are required, so please install them into your system.

```bash
numpy
scipy
sauvola  #(We provide this so do not worry about this)
```

## Usages

```python
import diqa

image_path ="PATH/TO/IMAGE"

# Each measure returns 'float value' rounded to 3 decimal points
diqa.estimate_skew(image_path)         # to measure skewness
diqa.estimate_contrast(image_path)     # to measure contrast
diqa.estimate_rangeeffect(image_path)  # to measure range-effect
diqa.estimate_bleedthrough(image_path) # to measure bleed-through
```

## License

***
# Page2dom
Will be updated soon...