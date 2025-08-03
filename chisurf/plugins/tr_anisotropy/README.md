# TR Anisotropy Plugin

This plugin provides tools for analyzing time-resolved fluorescence anisotropy data.

## Implementation Notes

- Uses ChiSurf's logging system instead of print statements for better log management
- Logs important events such as loading/saving settings and error conditions

## Features

- Load and process polarization-resolved fluorescence decay data
- Set up and visualize rotation spectra and lifetime components
- Create and manage anisotropy fits with multiple rotation correlation times
- Analyze rotational diffusion of fluorophores in different environments
- Intelligent background region selection that automatically sets the initial region to 30%-80% of the data range, optimized for typical IRF profiles
- Enhanced visualization with prominent background-corrected IRF (thicker lines) and semi-transparent (60% alpha) non-corrected IRF for better visual distinction
- Interactive legend that clearly identifies VV/VH and raw/corrected IRF curves

## User Settings

The plugin stores user-specific settings in the ChiSurf user settings directory:

```
<user_settings_path>/plugins/tr_anisotropy/
```

### wizard.spk.json

This file contains default lifetime and rotation spectrum settings for the anisotropy wizard. When the wizard is first opened, it:

1. Checks if wizard.spk.json exists in the user settings directory
2. If it doesn't exist, copies the default file from the plugin directory
3. Loads the settings from the file in the user settings directory
4. This ensures that user-specific settings are preserved between sessions

When saving settings:
1. The Save button saves directly to the current file without asking for a filename
2. Auto-save is performed when moving between wizard pages
3. If the file was previously saved to a different location than the default, it is also copied to the default location
4. A backup of the previous default file is created with the `.backup.json` extension

## Jordi Format Support

The plugin supports the Jordi format, which contains both VV and VH data in a single file. When `cs.current_setup.is_jordi = True`:

1. Only one file is required for IRF and one for data
2. The UI is updated to reflect this
3. The files are loaded with the appropriate polarization parameters