import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import yaml
import os
import chardet
import threading
import subprocess
import sys
import json
import tempfile

class CreateToolTip:
    """Create a tooltip for a given widget"""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind('<Enter>', self.enter)
        self.widget.bind('<Leave>', self.leave)

    def enter(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        
        label = ttk.Label(self.tooltip, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1)
        label.pack()

    def leave(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

class FluxTrainingGUI(ttk.Frame):
    def __init__(self, master=None, config=None):
        ttk.Frame.__init__(self, master)
        self.master = master
        self._raw_config = config if config is not None else {}
        self.current_config_path = None
        self.current_config_label_var = tk.StringVar(value='No config loaded')
        self.batch_configs = []  # List to store batch training configs
        
        # Configure window size and properties
        self.master.geometry("768x512")
        self.master.minsize(768, 512)
        
        # Settings file path
        self.settings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_settings.json")
        
        # Initialize learning rate presets
        self.lr_presets = {
            '1e-4 (Standard)': {'value': '1e-4', 'description': 'Standard learning rate for most cases'},
            '2e-4 (Faster)': {'value': '2e-4', 'description': 'Faster learning, may be less stable'},
            '5e-5 (Conservative)': {'value': '5e-5', 'description': 'More conservative, more stable'},
            '3e-5 (Very Conservative)': {'value': '3e-5', 'description': 'Very conservative, very stable'},
            '2e-5 (Minimal)': {'value': '2e-5', 'description': 'Minimal changes, highest stability'}
        }
        
        # Initialize containers
        self.step_buttons = []
        self.resolution_vars = {}
        
        # Initialize variables with defaults
        self.name_var = tk.StringVar(value='')
        self.trigger_word_var = tk.StringVar(value='')
        self.data_folder_var = tk.StringVar(value='')
        self.output_folder_var = tk.StringVar(value='')
        self.lr_var = tk.StringVar(value='1e-4')
        self.custom_lr_var = tk.StringVar()
        self.training_steps_var = tk.StringVar(value='1000')
        self.auto_steps_var = tk.BooleanVar(value=True)
        self.convert_utf8_var = tk.BooleanVar(value=True)
        self.prompt_style_var = tk.StringVar(value="Natural")
        self.sampling_steps_var = tk.StringVar(value="20")
        self.cfg_scale_var = tk.StringVar(value="3.5")
        self.save_dtype_var = tk.StringVar(value="float16")
        self.save_every_var = tk.StringVar(value="250")
        self.max_saves_var = tk.StringVar(value="4")
        self.model_path_var = tk.StringVar(value="")
        self.is_flux_var = tk.BooleanVar(value=True)
        self.quantize_var = tk.BooleanVar(value=True)
        self.dataset_stats_var = tk.StringVar(value='')
        self.enable_sampling_var = tk.BooleanVar(value=True)
        self.sample_every_var = tk.StringVar(value="500")
        
        # Network variables
        self.network_type_var = tk.StringVar(value="lora")
        self.linear_var = tk.StringVar(value="16")
        self.linear_alpha_var = tk.StringVar(value="16")
        
        # Training variables
        self.batch_size_var = tk.StringVar(value="1")
        self.steps_var = tk.StringVar(value="5000")
        self.grad_accum_var = tk.StringVar(value="1")
        self.train_unet_var = tk.BooleanVar(value=True)
        self.train_text_encoder_var = tk.BooleanVar(value=False)
        self.grad_checkpointing_var = tk.BooleanVar(value=True)
        self.noise_scheduler_var = tk.StringVar(value="flowmatch")
        self.optimizer_var = tk.StringVar(value="adamw8bit")
        
        # Create scrollable frame
        self.canvas = tk.Canvas(self)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Pack scrollbar and canvas
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Pack the main frame
        self.pack(fill="both", expand=True)
        
        # Bind mouse wheel to canvas
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)
        
        # Create widgets
        self.create_widgets()
        
        # Load last settings if available
        self.load_last_settings()
        
        # Then load config if provided (overrides last settings)
        if config:
            self.load_config(config)
            
        # Save settings when window is closed
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Add trace to data_folder_var to update stats when changed
        self.data_folder_var.trace_add("write", lambda *args: self.update_dataset_stats(self.data_folder_var.get()))

    def load_config(self, config):
        """Load configuration values into GUI elements"""
        try:
            if 'config' in config:
                # Basic settings
                self.name_var.set(config['config'].get('name', ''))
                self.trigger_word_var.set(config['config'].get('trigger_word', ''))
                
                # Process settings
                if 'process' in config['config'] and config['config']['process']:
                    process = config['config']['process'][0]
                    
                    # Model settings
                    if 'model' in process:
                        model = process['model']
                        self.model_path_var.set(model.get('name_or_path', ''))
                        self.is_flux_var.set(model.get('is_flux', True))
                        self.quantize_var.set(model.get('quantize', True))
                    
                    # Training settings
                    if 'train' in process:
                        train = process['train']
                        self.steps_var.set(str(train.get('steps', 5000)))
                        self.batch_size_var.set(str(train.get('batch_size', 1)))
                        self.grad_accum_var.set(str(train.get('gradient_accumulation_steps', 1)))
                        self.train_unet_var.set(train.get('train_unet', True))
                        self.train_text_encoder_var.set(train.get('train_text_encoder', False))
                        self.grad_checkpointing_var.set(train.get('gradient_checkpointing', True))
                        self.noise_scheduler_var.set(train.get('noise_scheduler', 'flowmatch'))
                        self.optimizer_var.set(train.get('optimizer', 'adamw8bit'))
                        
                        # Learning rate
                        lr = train.get('lr', '1e-4')
                        for preset_name, preset_data in self.lr_presets.items():
                            if preset_data['value'] == str(lr):
                                self.lr_var.set(preset_name)
                                break
                        else:
                            self.lr_var.set(str(lr))
                    
                    # Network settings
                    if 'network' in process:
                        network = process['network']
                        self.network_type_var.set(network.get('type', 'lora'))
                        self.linear_var.set(str(network.get('linear', 16)))
                        self.linear_alpha_var.set(str(network.get('linear_alpha', 16)))
                    
                    # Save settings
                    if 'save' in process:
                        save = process['save']
                        self.save_dtype_var.set(save.get('dtype', 'float16'))
                        self.save_every_var.set(str(save.get('save_every', 250)))
                        self.max_saves_var.set(str(save.get('max_step_saves_to_keep', 4)))
                    
                    # Dataset settings
                    if 'datasets' in process and process['datasets']:
                        dataset = process['datasets'][0]
                        folder_path = dataset.get('folder_path', '')
                        self.data_folder_var.set(folder_path)
                        
                        # Update dataset stats
                        self.update_dataset_stats(folder_path)
                        
                        # Resolution settings
                        resolutions = dataset.get('resolution', [])
                        for res in self.resolution_vars:
                            size = int(res.split('x')[0])
                            self.resolution_vars[res]['var'].set(size in resolutions)
                    
                    # Training folder
                    self.output_folder_var.set(process.get('training_folder', ''))
                    
                    # Sample settings
                    if 'sample' in process:
                        sample = process['sample']
                        self.enable_sampling_var.set(sample is not None)
                        if sample:
                            self.sample_every_var.set(str(sample.get('sample_every', 500)))
                            self.sampling_steps_var.set(str(sample.get('sample_steps', 20)))
                            self.cfg_scale_var.set(str(sample.get('guidance_scale', 3.5)))
                            if 'prompts' in sample and sample['prompts']:
                                self.prompt_text.delete('1.0', tk.END)
                                self.prompt_text.insert('1.0', '\n\n'.join(sample['prompts']))
                    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")

    def get_config_value(self, *keys, default=None):
        """Safely get nested config value without recursion"""
        try:
            value = self._raw_config
            for key in keys:
                if isinstance(value, (dict, list)):
                    value = value[key] if isinstance(value, dict) else value[int(key)]
                else:
                    return default
            return value
        except (KeyError, IndexError, ValueError, TypeError):
            return default

    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        if event.num == 5 or event.delta < 0:  # Scroll down
            self.canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta > 0:  # Scroll up
            self.canvas.yview_scroll(-1, "units")

    def calculate_suggested_steps(self, image_count):
        """Calculate suggested training steps based on image count and learning rate"""
        if image_count <= 0:
            return 5000  # Default value
        
        # Get current learning rate value
        selected_lr = self.lr_var.get()
        lr_value = float(self.lr_presets[selected_lr]['value'] if selected_lr in self.lr_presets else selected_lr)
        
        # Base calculation: roughly 100 steps per image
        suggested = image_count * 100
        
        # Adjust steps based on learning rate
        # Use 1e-4 as the baseline learning rate
        lr_multiplier = 1e-4 / lr_value
        suggested = int(suggested * lr_multiplier)
        
        # Round to nearest 500
        suggested = round(suggested / 500) * 500
        
        # Set minimum and maximum values
        suggested = max(1000, min(suggested, 10000))
        
        return suggested

    def update_suggested_steps(self, image_count):
        """Update steps based on auto calculation"""
        if self.auto_steps_var.get():
            suggested = self.calculate_suggested_steps(image_count)
            self.steps_var.set(str(suggested))

    def browse_folder(self, folder_type):
        folder = filedialog.askdirectory()
        if folder:
            if folder_type == 'data':
                self.data_folder_var.set(folder)
                # Update dataset stats
                self.update_dataset_stats(folder)
            elif folder_type == 'output':
                self.output_folder_var.set(folder)
            elif folder_type == 'model':
                self.model_path_var.set(folder)

    def toggle_steps_entry(self):
        """Toggle the steps entry between auto and manual"""
        if self.auto_steps_var.get():
            # If switching to auto, update with suggested steps
            if self.data_folder_var.get():
                image_count, _ = self.count_dataset_files(self.data_folder_var.get())
                self.update_suggested_steps(image_count)
        self.steps_entry.configure(state='disabled' if self.auto_steps_var.get() else 'normal')

    def create_widgets(self):
        # Initialize default prompts first
        self.default_natural_prompt = """[trigger], a stunning female model in an opulent mansion living room, wearing an elegant evening gown, perfect lighting, intricate architectural details, crystal chandelier, ornate gold-framed mirrors, marble fireplace, luxurious velvet furniture, hardwood floors, detailed wall moldings, floor-to-ceiling windows with silk curtains, best quality, masterpiece, highly detailed, photorealistic, ultra-realistic, professional photography, ambient lighting, cinematic composition

[trigger], beautiful woman in designer dress, luxury penthouse suite, panoramic city views, modern designer furniture, ambient evening lighting, professional fashion photography, perfect makeup, styled hair, best quality, photorealistic

[trigger], elegant female model, haute couture gown, grand ballroom setting, crystal chandeliers, marble columns, ornate architecture, professional lighting, fashion magazine quality, ultra-realistic, highly detailed"""

        self.default_danbooru_prompt = """[trigger], 1girl, solo, masterpiece, best quality, photorealistic, ultra-detailed, elegant, evening_gown, mansion, luxury, chandelier, ornate, marble_floor, gold_trim, velvet_furniture, floor_to_ceiling_windows, perfect_lighting, professional_photography

[trigger], 1girl, solo, beautiful, designer_dress, penthouse, city_view, modern_interior, evening_lighting, fashion_photography, perfect_makeup, styled_hair, high_quality, realistic

[trigger], 1girl, solo, glamour, haute_couture, ballroom, crystal_lighting, marble_columns, architectural, magazine_quality, ultra-realistic, highly_detailed"""

        self.default_t5_prompt = """Generate a photorealistic image of [trigger] as an elegant woman in a luxurious mansion living room. Include details of crystal chandeliers, marble floors, and ornate furniture. Style: professional photography, high quality.

Create a fashion photograph of [trigger] in a modern penthouse with city views. Show her in designer clothing with perfect makeup and lighting. Style: editorial photography.

Produce an ultra-realistic image of [trigger] in a grand ballroom setting with architectural details and luxury elements. Style: high-end fashion photography."""

        # Configure grid layout for scrollable frame
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(1, weight=1)
        current_row = 0

        # Create main notebook
        main_notebook = ttk.Notebook(self.scrollable_frame)
        main_notebook.grid(row=current_row, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)
        
        # Training tab
        training_tab = ttk.Frame(main_notebook)
        main_notebook.add(training_tab, text="Training")
        training_tab.grid_columnconfigure(0, weight=1)
        training_tab.grid_columnconfigure(1, weight=1)
        tab_row = 0

        # 1. Basic Configuration Section
        basic_frame = ttk.LabelFrame(training_tab, text="Basic Configuration")
        basic_frame.grid(row=tab_row, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)
        basic_frame.grid_columnconfigure(1, weight=1)
        
        # Name Entry
        ttk.Label(basic_frame, text="Name:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(basic_frame, textvariable=self.name_var).grid(row=0, column=1, columnspan=2, sticky='ew', padx=5, pady=2)
        
        # Trigger Word Entry
        ttk.Label(basic_frame, text="Trigger Word:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(basic_frame, textvariable=self.trigger_word_var).grid(row=1, column=1, columnspan=2, sticky='ew', padx=5, pady=2)
        
        # UTF-8 Conversion Checkbox
        utf8_checkbox = ttk.Checkbutton(basic_frame, text="Convert Captions to UTF-8", 
                                      variable=self.convert_utf8_var)
        utf8_checkbox.grid(row=2, column=0, columnspan=3, sticky='w', padx=5, pady=2)
        CreateToolTip(utf8_checkbox, "Convert dataset captions to UTF-8 encoding to avoid training errors")
        tab_row += 1

        # 2. Model Configuration
        model_frame = ttk.LabelFrame(training_tab, text="Model Configuration")
        model_frame.grid(row=tab_row, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)
        model_frame.grid_columnconfigure(1, weight=1)
        
        # Model Path
        model_path_frame = ttk.Frame(model_frame)
        model_path_frame.grid(row=0, column=0, columnspan=2, sticky='ew', padx=5, pady=2)
        ttk.Label(model_path_frame, text="Model Path:").pack(side=tk.LEFT)
        ttk.Entry(model_path_frame, textvariable=self.model_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(model_path_frame, text="Browse", command=lambda: self.browse_folder('model')).pack(side=tk.LEFT, padx=5)
        
        # Model Options
        model_options_frame = ttk.Frame(model_frame)
        model_options_frame.grid(row=1, column=0, columnspan=2, sticky='ew', padx=5, pady=2)
        ttk.Checkbutton(model_options_frame, text="Is Flux", variable=self.is_flux_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(model_options_frame, text="Quantize", variable=self.quantize_var).pack(side=tk.LEFT, padx=5)
        tab_row += 1

        # 3. Folders Configuration
        folders_frame = ttk.LabelFrame(training_tab, text="Folders")
        folders_frame.grid(row=tab_row, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)
        folders_frame.grid_columnconfigure(1, weight=1)
        
        # Output Folder
        ttk.Label(folders_frame, text="Output Folder:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(folders_frame, textvariable=self.output_folder_var).grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        ttk.Button(folders_frame, text="Browse", command=lambda: self.browse_folder("output")).grid(row=0, column=2, padx=5, pady=2)
        
        # Dataset Folder
        ttk.Label(folders_frame, text="Dataset Folder:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(folders_frame, textvariable=self.data_folder_var).grid(row=1, column=1, sticky='ew', padx=5, pady=2)
        ttk.Button(folders_frame, text="Browse", command=lambda: self.browse_folder('data')).grid(row=1, column=2, padx=5, pady=2)
        
        # Dataset Stats Label
        ttk.Label(folders_frame, textvariable=self.dataset_stats_var).grid(row=2, column=0, columnspan=3, sticky='w', padx=5, pady=2)
        
        tab_row += 1

        # 4. Network Configuration (Left Column)
        network_frame = ttk.LabelFrame(training_tab, text="Network Configuration")
        network_frame.grid(row=tab_row, column=0, sticky='nsew', padx=5, pady=5)
        network_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(network_frame, text="Linear:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(network_frame, textvariable=self.linear_var, width=8).grid(row=0, column=1, sticky='w', padx=5, pady=2)
        ttk.Label(network_frame, text="Alpha:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(network_frame, textvariable=self.linear_alpha_var, width=8).grid(row=1, column=1, sticky='w', padx=5, pady=2)

        # 5. Save Configuration (Right Column)
        save_frame = ttk.LabelFrame(training_tab, text="Save Configuration")
        save_frame.grid(row=tab_row, column=1, sticky='nsew', padx=5, pady=5)
        save_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(save_frame, text="Save Precision:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        ttk.OptionMenu(save_frame, self.save_dtype_var, "float16", "float16", "float32").grid(row=0, column=1, sticky='w', padx=5, pady=2)
        ttk.Label(save_frame, text="Save Every:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(save_frame, textvariable=self.save_every_var, width=8).grid(row=1, column=1, sticky='w', padx=5, pady=2)
        ttk.Label(save_frame, text="Max Saves:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(save_frame, textvariable=self.max_saves_var, width=8).grid(row=2, column=1, sticky='w', padx=5, pady=2)
        tab_row += 1

        # 6. Training Configuration (Left Column)
        train_frame = ttk.LabelFrame(training_tab, text="Training Configuration")
        train_frame.grid(row=tab_row, column=0, sticky='nsew', padx=5, pady=5)
        train_frame.grid_columnconfigure(1, weight=1)
        
        # Batch Size
        ttk.Label(train_frame, text="Batch Size:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(train_frame, textvariable=self.batch_size_var, width=10).grid(row=0, column=1, sticky='w', padx=5, pady=2)
        
        # Steps with Auto-Calculate Option
        ttk.Label(train_frame, text="Steps:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        steps_frame = ttk.Frame(train_frame)
        steps_frame.grid(row=1, column=1, sticky='w', padx=5, pady=2)
        
        self.steps_entry = ttk.Entry(steps_frame, textvariable=self.steps_var, width=10, state='disabled')
        self.steps_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        auto_steps_cb = ttk.Checkbutton(steps_frame, text="Auto", variable=self.auto_steps_var, 
                                      command=self.toggle_steps_entry)
        auto_steps_cb.pack(side=tk.LEFT)
        CreateToolTip(auto_steps_cb, "Automatically calculate steps based on dataset size (100 steps per image)")
        
        # Gradient Accumulation
        ttk.Label(train_frame, text="Gradient Accumulation:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(train_frame, textvariable=self.grad_accum_var, width=10).grid(row=2, column=1, sticky='w', padx=5, pady=2)
        
        # Learning Rate Dropdown
        ttk.Label(train_frame, text="Learning Rate:").grid(row=3, column=0, sticky='w', padx=5, pady=2)
        lr_menu = ttk.OptionMenu(train_frame, self.lr_var, list(self.lr_presets.keys())[0], *self.lr_presets.keys(),
                               command=lambda _: self.update_suggested_steps(self.count_dataset_files(self.data_folder_var.get())[0]))
        lr_menu.grid(row=3, column=1, sticky='w', padx=5, pady=2)
        
        # Create tooltip for learning rate
        def update_lr_tooltip(event=None):
            selected = self.lr_var.get()
            if selected in self.lr_presets:
                CreateToolTip(lr_menu, self.lr_presets[selected]['description'])
        
        lr_menu.bind('<Enter>', update_lr_tooltip)
        
        # Training Checkboxes
        ttk.Checkbutton(train_frame, text="Train U-Net", variable=self.train_unet_var).grid(row=4, column=0, sticky='w', padx=5)
        ttk.Checkbutton(train_frame, text="Train Text Encoder", variable=self.train_text_encoder_var).grid(row=5, column=0, sticky='w', padx=5)
        ttk.Checkbutton(train_frame, text="Gradient Checkpointing", variable=self.grad_checkpointing_var).grid(row=6, column=0, sticky='w', padx=5)
        
        ttk.Label(train_frame, text="Noise Scheduler:").grid(row=7, column=0, sticky='w', padx=5, pady=2)
        ttk.OptionMenu(train_frame, self.noise_scheduler_var, "flowmatch", "flowmatch").grid(row=7, column=1, sticky='w', padx=5, pady=2)
        ttk.Label(train_frame, text="Optimizer:").grid(row=8, column=0, sticky='w', padx=5, pady=2)
        ttk.OptionMenu(train_frame, self.optimizer_var, "adamw8bit", "adamw8bit").grid(row=8, column=1, sticky='w', padx=5, pady=2)

        # 7. Resolution Settings (Right Column)
        res_frame = ttk.LabelFrame(training_tab, text="Resolution Settings")
        res_frame.grid(row=tab_row, column=1, sticky='nsew', padx=5, pady=5)
        res_frame.grid_columnconfigure(0, weight=1)
        
        for i, size in enumerate([512, 768, 1024, 1280, 1536]):
            var = tk.BooleanVar(value=False)
            self.resolution_vars[f"{size}x{size}"] = {'var': var, 'size': size}
            ttk.Checkbutton(res_frame, text=f"{size}x{size}",
                          variable=var).grid(row=i//2, column=i%2, padx=5, pady=2, sticky='w')
        tab_row += 1

        # Sample tab
        sample_tab = ttk.Frame(main_notebook)
        main_notebook.add(sample_tab, text="Sample")
        sample_tab.grid_columnconfigure(0, weight=1)
        
        # Style selection
        style_frame = ttk.Frame(sample_tab)
        style_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=2)
        ttk.Label(style_frame, text="Prompt Style:").pack(side=tk.LEFT)
        for style in ["Natural", "Danbooru", "T5"]:
            ttk.Radiobutton(style_frame, text=style, variable=self.prompt_style_var, 
                          value=style, command=self.update_prompt_style).pack(side=tk.LEFT, padx=5)

        # Sample notebook for prompts and settings
        sample_notebook = ttk.Notebook(sample_tab)
        sample_notebook.grid(row=1, column=0, sticky='nsew', padx=5, pady=2)
        
        # Prompts tab
        prompts_tab = ttk.Frame(sample_notebook)
        sample_notebook.add(prompts_tab, text="Prompts")
        prompts_tab.grid_columnconfigure(0, weight=1)
        
        # Template notebook
        template_notebook = ttk.Notebook(prompts_tab)
        template_notebook.grid(row=0, column=0, sticky='nsew', padx=5, pady=2)
        
        # Create template categories and buttons
        categories = {
            'Women': [
                ('Elegant', '[trigger], elegant woman in designer evening gown, luxury mansion setting, professional lighting'),
                ('Business', '[trigger], confident businesswoman in tailored suit, modern office setting, professional lighting'),
                ('Casual', '[trigger], natural beauty in casual attire, outdoor setting, natural lighting'),
                ('Glamour', '[trigger], glamorous woman, perfect makeup, styled hair, luxury setting'),
                ('Fashion', '[trigger], high fashion model, designer clothing, studio lighting, magazine quality'),
                ('Portrait', '[trigger], sophisticated portrait, perfect lighting, professional photography')
            ],
            'Men': [
                ('Executive', '[trigger], distinguished businessman in tailored suit, luxury office setting'),
                ('Casual', '[trigger], handsome man in casual wear, natural setting'),
                ('Fashion', '[trigger], male model in designer clothing, studio lighting'),
                ('Sport', '[trigger], athletic man in fitness attire, gym setting'),
                ('Portrait', '[trigger], professional male portrait, business attire, studio lighting'),
                ('Outdoor', '[trigger], adventurous man, outdoor lifestyle, natural lighting')
            ],
            'Places': [
                ('Mansion', '[trigger], opulent mansion interior, crystal chandeliers, marble floors'),
                ('Modern', '[trigger], contemporary luxury apartment, city skyline view'),
                ('Nature', '[trigger], breathtaking landscape, majestic mountains, sunset'),
                ('Urban', '[trigger], vibrant cityscape, modern architecture, city lights'),
                ('Studio', '[trigger], professional photography studio, perfect lighting setup'),
                ('Office', '[trigger], luxury corporate office, modern design, city view')
            ],
            'Things': [
                ('Car', '[trigger], luxury sports car, studio lighting, showroom setting'),
                ('Fashion', '[trigger], luxury designer product, professional photography'),
                ('Tech', '[trigger], modern technology device, minimalist setting'),
                ('Art', '[trigger], fine art object, gallery setting, perfect lighting'),
                ('Jewelry', '[trigger], luxury jewelry piece, studio lighting, black background'),
                ('Product', '[trigger], premium product photography, professional lighting')
            ]
        }
        
        # Create tabs for each category
        for category, templates in categories.items():
            tab = ttk.Frame(template_notebook)
            template_notebook.add(tab, text=category)
            
            # Create buttons for templates in a grid
            for i, (name, prompt) in enumerate(templates):
                ttk.Button(tab, 
                          text=name,
                          command=lambda p=prompt: self.add_template_prompt(p)).grid(
                              row=i//3, column=i%3, padx=5, pady=2, sticky='w')

        # Text area with scrollbar
        text_container = ttk.Frame(prompts_tab)
        text_container.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        text_container.grid_columnconfigure(0, weight=1)
        text_container.grid_rowconfigure(0, weight=1)
        
        self.prompt_text = tk.Text(text_container, height=8, width=70, wrap=tk.WORD)
        self.prompt_text.grid(row=0, column=0, sticky='nsew')
        
        prompt_scrollbar = ttk.Scrollbar(text_container, orient="vertical", command=self.prompt_text.yview)
        prompt_scrollbar.grid(row=0, column=1, sticky='ns')
        self.prompt_text.configure(yscrollcommand=prompt_scrollbar.set)
        
        # Insert default text
        self.prompt_text.insert('1.0', self.default_natural_prompt)
        
        # Sample Settings tab
        settings_tab = ttk.Frame(sample_notebook)
        sample_notebook.add(settings_tab, text="Settings")
        settings_tab.grid_columnconfigure(1, weight=1)
        
        # Enable/Disable sampling
        self.enable_sampling_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_tab, text="Enable Sampling", 
                       variable=self.enable_sampling_var).grid(row=0, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        
        # Sample frequency
        ttk.Label(settings_tab, text="Sample Every:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.sample_every_var = tk.StringVar(value="500")
        ttk.Entry(settings_tab, textvariable=self.sample_every_var, width=10).grid(row=1, column=1, sticky='w', padx=5, pady=2)
        CreateToolTip(settings_tab.winfo_children()[-1], "Generate samples every N steps (0 to disable)")
        
        # Sample settings
        ttk.Label(settings_tab, text="Sampling Steps:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(settings_tab, textvariable=self.sampling_steps_var, width=10).grid(row=2, column=1, sticky='w', padx=5, pady=2)
        CreateToolTip(settings_tab.winfo_children()[-1], "Number of steps for sample generation")

        ttk.Label(settings_tab, text="CFG Scale:").grid(row=3, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(settings_tab, textvariable=self.cfg_scale_var, width=10).grid(row=3, column=1, sticky='w', padx=5, pady=2)
        CreateToolTip(settings_tab.winfo_children()[-1], "Classifier Free Guidance scale for sample generation")

        # Add Batch Training tab
        batch_tab = ttk.Frame(main_notebook)
        main_notebook.add(batch_tab, text="Batch Training")
        batch_tab.grid_columnconfigure(0, weight=1)
        
        # Batch list frame
        batch_frame = ttk.LabelFrame(batch_tab, text="Training Queue")
        batch_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        batch_frame.grid_columnconfigure(0, weight=1)
        
        # Create treeview for batch list
        self.batch_tree = ttk.Treeview(batch_frame, columns=('Name', 'Path'), show='headings', height=10)
        self.batch_tree.heading('Name', text='Config Name')
        self.batch_tree.heading('Path', text='Config Path')
        self.batch_tree.column('Name', width=150)
        self.batch_tree.column('Path', width=400)
        self.batch_tree.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        
        # Scrollbar for treeview
        batch_scroll = ttk.Scrollbar(batch_frame, orient="vertical", command=self.batch_tree.yview)
        batch_scroll.grid(row=0, column=1, sticky='ns')
        self.batch_tree.configure(yscrollcommand=batch_scroll.set)
        
        # Buttons frame
        batch_buttons_frame = ttk.Frame(batch_tab)
        batch_buttons_frame.grid(row=1, column=0, sticky='ew', padx=5, pady=5)
        
        ttk.Button(batch_buttons_frame, text="Add Current to Batch", 
                  command=self.add_to_batch).pack(side=tk.LEFT, padx=5)
        ttk.Button(batch_buttons_frame, text="Add Config File", 
                  command=self.add_config_to_batch).pack(side=tk.LEFT, padx=5)
        ttk.Button(batch_buttons_frame, text="Remove Selected", 
                  command=self.remove_from_batch).pack(side=tk.LEFT, padx=5)
        ttk.Button(batch_buttons_frame, text="Clear Batch", 
                  command=self.clear_batch).pack(side=tk.LEFT, padx=5)
        ttk.Button(batch_buttons_frame, text="Start Batch Training", 
                  command=self.start_batch_training).pack(side=tk.RIGHT, padx=5)

        # Buttons at the bottom
        buttons_frame = ttk.Frame(self.scrollable_frame)
        buttons_frame.grid(row=current_row + 1, column=0, columnspan=2, sticky='ew', padx=5, pady=10)
        buttons_frame.grid_columnconfigure(1, weight=1)
        
        # Config section on the left
        config_section = ttk.Frame(buttons_frame)
        config_section.grid(row=0, column=0, sticky='w')
        
        # Config buttons
        config_buttons_frame = ttk.Frame(config_section)
        config_buttons_frame.pack(side=tk.TOP, anchor='w')
        
        load_config_btn = ttk.Button(config_buttons_frame, text="Load Config", command=self.load_config_file)
        load_config_btn.pack(side=tk.LEFT, padx=5)
        
        save_config_btn = ttk.Button(config_buttons_frame, text="Save Config", command=self.save_config)
        save_config_btn.pack(side=tk.LEFT, padx=5)
        
        update_config_btn = ttk.Button(config_buttons_frame, text="Update Config", command=self.update_config)
        update_config_btn.pack(side=tk.LEFT, padx=5)
        
        # Config label
        config_label = ttk.Label(config_section, textvariable=self.current_config_label_var, 
                               font=('TkDefaultFont', 8), foreground='gray')
        config_label.pack(side=tk.TOP, anchor='w', padx=5, pady=(2, 0))
        
        # Start Training button on the right
        start_training_btn = ttk.Button(buttons_frame, text="Start Training", command=self.start_training)
        start_training_btn.grid(row=0, column=2, padx=5)

    def count_dataset_files(self, folder):
        """Count the number of images and caption files in the dataset folder"""
        image_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
        caption_extensions = {'.txt', '.caption'}
        
        image_count = 0
        caption_count = 0
        
        try:
            for file in os.listdir(folder):
                lower_file = file.lower()
                ext = os.path.splitext(lower_file)[1]
                if ext in image_extensions:
                    image_count += 1
                elif ext in caption_extensions:
                    caption_count += 1
            
            return image_count, caption_count
        except Exception as e:
            messagebox.showerror("Error", f"Failed to count files: {str(e)}")
            return 0, 0

    def update_prompt_style(self):
        """Update the prompt text based on selected style"""
        style = self.prompt_style_var.get()
        self.prompt_text.delete('1.0', tk.END)
        
        if style == "Natural":
            self.prompt_text.insert('1.0', self.default_natural_prompt)
        elif style == "Danbooru":
            self.prompt_text.insert('1.0', self.default_danbooru_prompt)
        elif style == "T5":
            self.prompt_text.insert('1.0', self.default_t5_prompt)

    def add_template_prompt(self, prompt):
        """Add template prompt to the text area"""
        # Add newline if there's already text
        if self.prompt_text.get('1.0', tk.END).strip():
            self.prompt_text.insert(tk.END, '\n\n')
        self.prompt_text.insert(tk.END, prompt)

    def prepare_config(self):
        """Prepare the configuration dictionary"""
        # Get the actual learning rate value from the preset
        selected_lr = self.lr_var.get()
        lr_value = self.lr_presets[selected_lr]['value'] if selected_lr in self.lr_presets else selected_lr
        
        # Get selected resolutions
        selected_resolutions = [res for res, data in self.resolution_vars.items() if data['var'].get()]
        
        # Prepare sample config
        sample_config = None
        if self.enable_sampling_var.get():
            # Split prompts by blank lines and remove empty strings
            prompts = [p.strip() for p in self.prompt_text.get('1.0', tk.END).split('\n\n')]
            prompts = [p for p in prompts if p]  # Remove empty prompts
            
            sample_config = {
                'sampler': 'flowmatch',  # Must match train.noise_scheduler
                'sample_every': int(self.sample_every_var.get()),
                'width': 1024,  # Default size
                'height': 1024,  # Default size
                'prompts': prompts,
                'neg': '',  # Not used for flux
                'seed': 42,
                'walk_seed': True,
                'guidance_scale': float(self.cfg_scale_var.get()),
                'sample_steps': int(self.sampling_steps_var.get())
            }
        
        return {
            'job': 'extension',
            'config': {
                'name': self.name_var.get(),
                'trigger_word': self.trigger_word_var.get(),
                'process': [{
                    'type': 'sd_trainer',
                    'training_folder': self.output_folder_var.get(),
                    'device': 'cuda:0',
                    'network': {
                        'type': self.network_type_var.get(),
                        'linear': int(self.linear_var.get()),
                        'linear_alpha': int(self.linear_alpha_var.get())
                    },
                    'save': {
                        'dtype': self.save_dtype_var.get(),
                        'save_every': int(self.save_every_var.get()),
                        'max_step_saves_to_keep': int(self.max_saves_var.get()),
                        'push_to_hub': False
                    },
                    'datasets': [{
                        'folder_path': self.data_folder_var.get(),
                        'caption_ext': 'txt',
                        'caption_dropout_rate': 0.05,
                        'shuffle_tokens': False,
                        'cache_latents_to_disk': True,
                        'resolution': [int(res.split('x')[0]) for res, data in self.resolution_vars.items() if data['var'].get()]
                    }],
                    'train': {
                        'batch_size': int(self.batch_size_var.get()),
                        'steps': int(self.steps_var.get()),
                        'gradient_accumulation_steps': int(self.grad_accum_var.get()),
                        'train_unet': self.train_unet_var.get(),
                        'train_text_encoder': self.train_text_encoder_var.get(),
                        'gradient_checkpointing': self.grad_checkpointing_var.get(),
                        'noise_scheduler': self.noise_scheduler_var.get(),
                        'optimizer': self.optimizer_var.get(),
                        'lr': lr_value,
                        'ema_config': {
                            'use_ema': True,
                            'ema_decay': 0.99
                        },
                        'dtype': 'bf16'
                    },
                    'model': {
                        'name_or_path': self.model_path_var.get(),
                        'is_flux': self.is_flux_var.get(),
                        'quantize': self.quantize_var.get()
                    },
                    'sample': sample_config
                }]
            },
            'meta': {
                'name': '[name]',
                'version': '1.0'
            }
        }

    def save_config(self):
        """Save the current configuration to a YAML file"""
        config = self.prepare_config()
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                self.current_config_path = file_path
                self.current_config_label_var.set(f"Loaded: {os.path.basename(file_path)}")
                messagebox.showinfo("Success", "Configuration saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")

    def start_training(self):
        """Start the training process"""
        try:
            config = self.prepare_config()

            # Create output folder if it doesn't exist
            os.makedirs(self.output_folder_var.get(), exist_ok=True)
            
            # Use the current config file if one is loaded, otherwise create a temporary one
            if self.current_config_path:
                config_path = self.current_config_path
                # Update the current config file
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            else:
                # Create a temporary config file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_config:
                    yaml.dump(config, temp_config, default_flow_style=False, sort_keys=False)
                    config_path = temp_config.name
            
            # Get the path to run.py in the same directory as the GUI script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            train_script = os.path.join(script_dir, "run.py")
            
            # Start training in a separate process - pass config path as positional argument
            process = subprocess.Popen([sys.executable, train_script, config_path])
            
            # If using a temporary config, set up cleanup
            if not self.current_config_path:
                def cleanup_temp_config():
                    process.wait()
                    try:
                        os.unlink(config_path)
                    except:
                        pass  # Ignore errors in cleanup
                
                # Start cleanup thread
                cleanup_thread = threading.Thread(target=cleanup_temp_config)
                cleanup_thread.daemon = True
                cleanup_thread.start()
            
            messagebox.showinfo("Success", f"Training started with config: {os.path.basename(config_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start training: {str(e)}")

    def save_last_settings(self):
        """Save current settings to JSON file"""
        settings = {
            'name': self.name_var.get(),
            'trigger_word': self.trigger_word_var.get(),
            'data_folder': self.data_folder_var.get(),
            'output_folder': self.output_folder_var.get(),
            'model_path': self.model_path_var.get(),
            'is_flux': self.is_flux_var.get(),
            'quantize': self.quantize_var.get(),
            'convert_utf8': self.convert_utf8_var.get(),
            'learning_rate': self.lr_var.get(),
            'batch_size': self.batch_size_var.get(),
            'steps': self.steps_var.get(),
            'auto_steps': self.auto_steps_var.get(),
            'grad_accum': self.grad_accum_var.get(),
            'train_unet': self.train_unet_var.get(),
            'train_text_encoder': self.train_text_encoder_var.get(),
            'grad_checkpointing': self.grad_checkpointing_var.get(),
            'noise_scheduler': self.noise_scheduler_var.get(),
            'optimizer': self.optimizer_var.get(),
            'save_dtype': self.save_dtype_var.get(),
            'save_every': self.save_every_var.get(),
            'max_saves': self.max_saves_var.get(),
            'network_type': self.network_type_var.get(),
            'linear': self.linear_var.get(),
            'linear_alpha': self.linear_alpha_var.get(),
            'sampling_steps': self.sampling_steps_var.get(),
            'cfg_scale': self.cfg_scale_var.get(),
            'prompt_style': self.prompt_style_var.get(),
            'prompt_text': self.prompt_text.get('1.0', tk.END).strip(),
            'resolutions': {size: data['var'].get() for size, data in self.resolution_vars.items()},
            'batch_configs': self.batch_configs  # Save batch configs
        }
        
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            print(f"Failed to save settings: {str(e)}")

    def load_last_settings(self):
        """Load last used settings from JSON file"""
        if not os.path.exists(self.settings_file):
            return
            
        try:
            with open(self.settings_file, 'r') as f:
                settings = json.load(f)
                
            # Load all the settings
            self.name_var.set(settings.get('name', ''))
            self.trigger_word_var.set(settings.get('trigger_word', ''))
            self.data_folder_var.set(settings.get('data_folder', ''))
            self.output_folder_var.set(settings.get('output_folder', ''))
            self.model_path_var.set(settings.get('model_path', ''))
            self.is_flux_var.set(settings.get('is_flux', True))
            self.quantize_var.set(settings.get('quantize', True))
            self.convert_utf8_var.set(settings.get('convert_utf8', True))
            self.lr_var.set(settings.get('learning_rate', '1e-4'))
            self.batch_size_var.set(settings.get('batch_size', '1'))
            self.steps_var.set(settings.get('steps', '5000'))
            self.auto_steps_var.set(settings.get('auto_steps', True))
            self.grad_accum_var.set(settings.get('grad_accum', '1'))
            self.train_unet_var.set(settings.get('train_unet', True))
            self.train_text_encoder_var.set(settings.get('train_text_encoder', False))
            self.grad_checkpointing_var.set(settings.get('grad_checkpointing', True))
            self.noise_scheduler_var.set(settings.get('noise_scheduler', 'flowmatch'))
            self.optimizer_var.set(settings.get('optimizer', 'adamw8bit'))
            self.save_dtype_var.set(settings.get('save_dtype', 'float16'))
            self.save_every_var.set(settings.get('save_every', '250'))
            self.max_saves_var.set(settings.get('max_saves', '4'))
            self.network_type_var.set(settings.get('network_type', 'lora'))
            self.linear_var.set(settings.get('linear', '16'))
            self.linear_alpha_var.set(settings.get('linear_alpha', '16'))
            self.sampling_steps_var.set(settings.get('sampling_steps', '20'))
            self.cfg_scale_var.set(settings.get('cfg_scale', '3.5'))
            self.prompt_style_var.set(settings.get('prompt_style', 'Natural'))
            
            # Load prompt text
            if 'prompt_text' in settings:
                self.prompt_text.delete('1.0', tk.END)
                self.prompt_text.insert('1.0', settings['prompt_text'])
            
            # Load resolutions
            if 'resolutions' in settings:
                for size, value in settings['resolutions'].items():
                    if size in self.resolution_vars:
                        self.resolution_vars[size]['var'].set(value)
                        
            # Update steps entry state
            if hasattr(self, 'steps_entry'):
                self.steps_entry.configure(state='disabled' if self.auto_steps_var.get() else 'normal')
            
            # Update dataset stats if data folder exists
            self.update_dataset_stats(settings.get('data_folder', ''))
            
            # Load batch configs
            self.batch_configs = settings.get('batch_configs', [])
            for config_path in self.batch_configs:
                if os.path.exists(config_path):
                    self.batch_tree.insert('', 'end', values=(os.path.basename(config_path), config_path))
                
        except Exception as e:
            print(f"Failed to load settings: {str(e)}")

    def on_closing(self):
        """Handle window closing"""
        self.save_last_settings()
        self.master.destroy()

    def load_config_file(self):
        """Load configuration from a YAML file"""
        file_path = filedialog.askopenfilename(
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                self.load_config(config)
                self.current_config_path = file_path
                self.current_config_label_var.set(f"Loaded: {os.path.basename(file_path)}")
                messagebox.showinfo("Success", "Configuration loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")

    def update_config(self):
        """Update the currently loaded config file"""
        if not self.current_config_path:
            messagebox.showwarning("Warning", "No configuration file loaded. Please load a config file first.")
            return
            
        try:
            config = self.prepare_config()
            
            with open(self.current_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            self.current_config_label_var.set(f"Updated: {os.path.basename(self.current_config_path)}")
            messagebox.showinfo("Success", "Configuration updated successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update configuration: {str(e)}")

    def update_dataset_stats(self, folder_path):
        """Update dataset statistics for a given folder path"""
        if folder_path and os.path.exists(folder_path):
            image_count, caption_count = self.count_dataset_files(folder_path)
            self.dataset_stats_var.set(f"Images: {image_count}, Captions: {caption_count}")
            if self.auto_steps_var.get():
                self.update_suggested_steps(image_count)

    def add_to_batch(self):
        """Add current configuration to batch"""
        if not self.current_config_path:
            messagebox.showwarning("Warning", "Please save or load a configuration first.")
            return
            
        # Check if config is already in batch
        for item in self.batch_tree.get_children():
            if self.batch_tree.item(item)['values'][1] == self.current_config_path:
                messagebox.showinfo("Info", "This configuration is already in the batch.")
                return
        
        # Add to treeview
        self.batch_tree.insert('', 'end', values=(os.path.basename(self.current_config_path), 
                                                 self.current_config_path))
        self.batch_configs.append(self.current_config_path)

    def add_config_to_batch(self):
        """Add configuration file to batch"""
        file_paths = filedialog.askopenfilenames(
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        
        for file_path in file_paths:
            # Check if config is already in batch
            if any(file_path == self.batch_tree.item(item)['values'][1] 
                  for item in self.batch_tree.get_children()):
                continue
                
            # Add to treeview
            self.batch_tree.insert('', 'end', values=(os.path.basename(file_path), file_path))
            self.batch_configs.append(file_path)

    def remove_from_batch(self):
        """Remove selected configuration from batch"""
        selected = self.batch_tree.selection()
        if not selected:
            return
            
        for item in selected:
            config_path = self.batch_tree.item(item)['values'][1]
            self.batch_configs.remove(config_path)
            self.batch_tree.delete(item)

    def clear_batch(self):
        """Clear all configurations from batch"""
        self.batch_tree.delete(*self.batch_tree.get_children())
        self.batch_configs.clear()

    def start_batch_training(self):
        """Start batch training process"""
        if not self.batch_configs:
            messagebox.showwarning("Warning", "Batch queue is empty. Please add configurations first.")
            return
            
        try:
            # Get the path to run.py in the same directory as the GUI script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            train_script = os.path.join(script_dir, "run.py")
            
            def run_batch():
                for config_path in self.batch_configs:
                    try:
                        # Start training process
                        process = subprocess.Popen([sys.executable, train_script, config_path])
                        process.wait()  # Wait for the current training to complete
                    except Exception as e:
                        print(f"Error running config {config_path}: {str(e)}")
                
                messagebox.showinfo("Success", "Batch training completed!")
            
            # Start batch training in a separate thread
            batch_thread = threading.Thread(target=run_batch)
            batch_thread.daemon = True
            batch_thread.start()
            
            messagebox.showinfo("Success", "Batch training started!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start batch training: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Flux Training GUI")
    
    # Center the window on screen
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - 768) // 2
    y = (screen_height - 512) // 2
    root.geometry(f"768x512+{x}+{y}")
    
    app = FluxTrainingGUI(root)
    root.mainloop() 