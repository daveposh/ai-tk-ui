import customtkinter as ctk
import tkinter as tk
import tkinter.ttk as ttk

class Themes:
    def __init__(self):
        self.themes = {
            'dark': {
                'bg': '#1e1e1e',            # Darker background
                'fg': '#dce4ee',            # Light gray text
                'text': '#ffffff',          # White text
                'frame': '#242424',         # Frame background
                'button': '#1f538d',        # Blue button color
                'highlight': '#144870',     # Darker blue for hover
                'entry': '#343638',         # Dark entry background
                'entry_text': '#ffffff',    # White entry text
                'dropdown': '#2b2b2b',       # Dark dropdown background
                'dropdown_hover': '#1f538d', # Darker blue for dropdown hover
                'checkbox': '#1f538d',      # Checkbox active color
                'checkbox_hover': '#144870', # Darker blue for checkbox hover
                'progressbar': '#1f538d',    # Progress bar color
                'slider': '#1f538d'         # Slider color
            },
            'light': {
                'bg': '#ebebeb',            # Light gray background
                'fg': '#1a1a1a',            # Dark text
                'text': '#000000',          # Black text
                'frame': '#ffffff',         # White frame background
                'button': '#3b8ed0',        # Blue button color
                'highlight': '#36719f',     # Darker blue for hover
                'entry': '#e9e9e9',          # Light entry background
                'entry_text': '#000000',    # Black entry text
                'dropdown': '#ffffff',       # Light dropdown background
                'dropdown_hover': '#3b8ed0', # Blue for dropdown hover
                'checkbox': '#3b8ed0',      # Checkbox active color
                'checkbox_hover': '#36719f', # Darker blue for checkbox hover
                'progressbar': '#3b8ed0',    # Progress bar color
                'slider': '#3b8ed0'         # Slider color
            }
        }

    def get_theme(self, is_dark):
        """Get the theme colors based on dark mode setting"""
        return self.themes['dark'] if is_dark else self.themes['light']

    def apply_theme(self, widget, is_dark):
        """Apply theme to widget and all its children"""
        theme = self.get_theme(is_dark)
        
        # Configure customtkinter appearance
        ctk.set_appearance_mode("dark" if is_dark else "light")
        ctk.set_default_color_theme("blue")
        
        if isinstance(widget, (ctk.CTkButton, ttk.Button)):
            widget.configure(
                fg_color=theme['button'],
                hover_color=theme['highlight'],
                text_color=theme['text']
            )
        elif isinstance(widget, ctk.CTkOptionMenu):
            widget.configure(
                fg_color=theme['button'],
                button_color=theme['button'],
                button_hover_color=theme['highlight'],
                dropdown_fg_color=theme['frame'],
                dropdown_hover_color=theme['highlight'],
                text_color=theme['text'],
                dropdown_text_color=theme['text']
            )
        elif isinstance(widget, ctk.CTkFrame):
            widget.configure(fg_color=theme['frame'])
        elif isinstance(widget, (ctk.CTkLabel, ttk.Label)):
            widget.configure(text_color=theme['text'])
        elif isinstance(widget, (ctk.CTkEntry, ttk.Entry)):
            widget.configure(
                fg_color=theme['entry'],
                text_color=theme['entry_text']
            )
        elif isinstance(widget, tk.Text):
            widget.configure(
                bg=theme['entry'],
                fg=theme['entry_text'],
                insertbackground=theme['text']
            )
        
        # Recursively apply theme to all children
        for child in widget.winfo_children():
            self.apply_theme(child, is_dark)