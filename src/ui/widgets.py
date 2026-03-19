import pygame

from src.core.settings import (
    COLOR_BORDER,
    COLOR_BTN,
    COLOR_BTN_ACTIVE,
    COLOR_SLIDER_BG,
    COLOR_SLIDER_FILL,
    COLOR_TEXT,
)


class Button:
    def __init__(self, rect, label):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(event.pos)
        return False

    def draw(self, surface, font):
        color = COLOR_BTN_ACTIVE if self.active else COLOR_BTN
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        pygame.draw.rect(surface, COLOR_BORDER, self.rect, 1, border_radius=6)
        text = font.render(self.label, True, (245, 245, 245))
        surface.blit(text, text.get_rect(center=self.rect.center))


class Slider:
    def __init__(
        self,
        rect,
        label,
        min_value,
        max_value,
        value,
        fmt="{:.2f}",
        integer=False,
    ):
        self.track = pygame.Rect(rect)
        self.label = label
        self.min_value = min_value
        self.max_value = max_value
        self.value = value
        self.fmt = fmt
        self.integer = integer
        self.dragging = False

    def _ratio(self):
        if self.max_value <= self.min_value:
            return 0.0
        return (self.value - self.min_value) / (self.max_value - self.min_value)

    def _set_from_x(self, x):
        ratio = (x - self.track.left) / max(self.track.width, 1)
        ratio = max(0.0, min(1.0, ratio))
        value = self.min_value + ratio * (self.max_value - self.min_value)
        self.value = int(round(value)) if self.integer else float(value)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.track.collidepoint(event.pos):
                self.dragging = True
                self._set_from_x(event.pos[0])
                return True
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._set_from_x(event.pos[0])
            return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        return False

    def draw(self, surface, font):
        pygame.draw.rect(surface, COLOR_SLIDER_BG, self.track, border_radius=5)
        filled = self.track.copy()
        filled.width = int(self.track.width * self._ratio())
        pygame.draw.rect(surface, COLOR_SLIDER_FILL, filled, border_radius=5)
        pygame.draw.rect(surface, COLOR_BORDER, self.track, 1, border_radius=5)

        text = f"{self.label}: {self.fmt.format(self.value)}"
        label_surface = font.render(text, True, COLOR_TEXT)
        surface.blit(label_surface, (self.track.left, self.track.top - 22))


class NumericInput:
    def __init__(self, rect, min_value, max_value, value, fmt="{:.2f}", integer=False):
        self.rect = pygame.Rect(rect)
        self.min_value = min_value
        self.max_value = max_value
        self.value = value
        self.fmt = fmt
        self.integer = integer

        self.active = False
        self.text = self._format_value(self.value)
        self._dirty = False

    def _format_value(self, value):
        if self.integer:
            return str(int(round(value)))
        return self.fmt.format(value)

    def set_value(self, value):
        self.value = value
        if not self.active:
            self.text = self._format_value(self.value)

    def _parse(self):
        raw = self.text.strip().replace(",", ".")
        if raw == "":
            return None

        try:
            value = int(raw) if self.integer else float(raw)
        except ValueError:
            return None

        if value < self.min_value:
            value = self.min_value
        if value > self.max_value:
            value = self.max_value

        if self.integer:
            value = int(round(value))
        return value

    def handle_event(self, event):
        committed = False
        value = None

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            was_active = self.active
            self.active = self.rect.collidepoint(event.pos)
            if was_active and not self.active:
                value = self._parse()
                if value is not None:
                    self.value = value
                    committed = True
                self.text = self._format_value(self.value)
                self._dirty = False

        elif event.type == pygame.KEYDOWN and self.active:
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                value = self._parse()
                if value is not None:
                    self.value = value
                    committed = True
                self.text = self._format_value(self.value)
                self.active = False
                self._dirty = False
            elif event.key == pygame.K_ESCAPE:
                self.text = self._format_value(self.value)
                self.active = False
                self._dirty = False
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
                self._dirty = True
            else:
                allowed = "0123456789"
                if not self.integer:
                    allowed += ".,"
                if event.unicode and event.unicode in allowed:
                    self.text += event.unicode
                    self._dirty = True

        return committed, value

    def draw(self, surface, font):
        bg = (255, 255, 255) if self.active else (245, 246, 249)
        pygame.draw.rect(surface, bg, self.rect, border_radius=5)
        pygame.draw.rect(surface, COLOR_BORDER, self.rect, 1, border_radius=5)

        shown = self.text if self.active or self._dirty else self._format_value(self.value)
        text_surface = font.render(shown, True, COLOR_TEXT)
        text_y = self.rect.y + (self.rect.height - text_surface.get_height()) // 2
        surface.blit(text_surface, (self.rect.x + 8, text_y))
