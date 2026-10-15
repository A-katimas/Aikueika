import colorsys
from random import choice

THEME_COLOR = {
    "green": (50, 255, 55, 255),
    "red": (230, 41, 55, 255),
    "blue": (50, 41, 230, 255),
    "yellow": (250, 250, 50, 255),
    "purple": (200, 122, 255, 255),
    "black": (0, 0, 0, 255),
    "brown": (127, 106, 79, 255),
    "orange": (255, 161, 0, 255),
    "maroon": (190, 33, 55, 255),
    "gold": (255, 203, 0, 255),
    "darkred": (139, 0, 0, 255),
    "violet": (135, 60, 190, 255),
    "crimson": (220, 20, 60, 255),
    "cyan": (43, 255, 255, 255),
    "deepbrown": (63, 31, 15, 255),
    "darkbrown": (31, 15, 7, 255),
    "teal": (0, 127, 127, 255),
    "white": (255, 255, 255, 255),
    "lightgrey": (191, 191, 191, 255),
    "grey": (95, 95, 95, 255),
    "darkgrey": (63, 63, 63, 255),
    "magenta": (207, 0, 207, 255),
    "beige": (255, 191, 127, 255),
    "lightbeige": (255, 255, 191, 255),
    "lime": (127, 255, 63, 255),
    "darklime": (63, 127, 31, 255),
    "lightbrown": (255, 127, 63, 255),
    "lightblue": (63, 127, 255, 255),
    "navy": (15, 31, 127, 255),
    "lightpurple": (191, 47, 191, 255),
    "transpa": (255, 255, 255, 10),
    "invyziblmepatropkanmem": (255, 255, 255, 75),
}


class Rainbow:
    """
    Dégradé chromatique : chaque appel avance sur le cercle des teintes.

    Args:
        degrees: Pas de changement de teinte par appel, en degrés (0-360).
                 1° = très fluide (360 appels pour un tour complet),
                 5° = visible, 20° = très contrasté.
        start: Teinte de départ en degrés.
        saturation: 0.0 (gris) à 1.0 (couleur pleine).
        value: 0.0 (noir) à 1.0 (lumineux).
    """

    def __init__(
        self,
        degrees: float = 2.0,
        start: float = 0.0,
        saturation: float = 1.0,
        value: float = 1.0,
    ):
        self.degrees = degrees
        self.start = start
        self.saturation = saturation
        self.value = value
        self.hue = start % 360

    def reset(self) -> None:
        self.hue = self.start % 360

    def give(self) -> list[int]:
        r, g, b = (
            round(c * 255)
            for c in colorsys.hsv_to_rgb(
                self.hue / 360, self.saturation, self.value
            )
        )
        self.hue = (self.hue + self.degrees) % 360
        return [r, g, b]

    def __call__(self, text: str | int) -> str:
        r, g, b = (
            round(c * 255)
            for c in colorsys.hsv_to_rgb(
                self.hue / 360, self.saturation, self.value
            )
        )
        self.hue = (self.hue + self.degrees) % 360
        return f"\033[38;2;{r};{g};{b}m{text}\033[0m"


def color(text: str | int | None, r: int, g: int, b: int) -> str:
    """
    Apply an RGB foreground color to the given text using ANSI escape codes.

    This function wraps the input text with ANSI codes to display it
    in the specified RGB color in the terminal.

    Args:
        text: The text (or value) to color. If not a string, it will be
        converted.
        r: Red component (0–255).
        g: Green component (0–255).
        b: Blue component (0–255).

    Returns:
        A string formatted with ANSI escape codes for colored output.
    """
    if isinstance(text, str):
        return f"\033[38;2;{r};{g};{b}m{text}\033[0m"
    return f"\033[38;2;{r};{g};{b}m{str(text)}\033[0m"


def bg_color(text: str | int, r: int, g: int, b: int) -> str:
    """
    Apply an RGB background color to the given text using ANSI escape codes.

    This function wraps the input text with ANSI codes to display it
    with the specified RGB background color in the terminal.

    Args:
        text: The text (or value) to display. If not a string, it will be
        converted.
        r: Red component (0–255).
        g: Green component (0–255).
        b: Blue component (0–255).

    Returns:
        A string formatted with ANSI escape codes for background-colored
        output.
    """
    if isinstance(text, str):
        return f"\033[48;2;{r};{g};{b}m{str(text)}\033[0m"
    return f"\033[48;2;{r};{g};{b}m{text}\033[0m"


def random_color(text: str | int) -> str:
    """
    Apply a random color from the THEME_COLOR palette to the given text.
    """
    rgb = choice(list(THEME_COLOR.values()))
    r, g, b = rgb[0], rgb[1], rgb[2]
    if isinstance(text, str):
        return f"\033[38;2;{r};{g};{b}m{str(text)}\033[0m"
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"


def chose_color(text: str, color: int) -> str:
    """
    Apply a specific color from the THEME_COLOR palette to the given text.
    """
    r, g, b, _ = list(THEME_COLOR.values())[color % len(THEME_COLOR)]
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"
