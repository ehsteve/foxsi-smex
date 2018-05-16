;+
; NAME : foxsi_load_optics_effective_area
;
; PURPOSE : Load the optics effective area file.
;
; SYNTAX : data = foxsi_load_optics_effarea()
;
; INPUTS : None
;
; Optional Inputs : None
;
; KEYWORDS :  None
;
; RETURNS : struct
;
; EXAMPLES : None
;

FUNCTION foxsi_load_optics_effective_area

    ; load the foxsi-smex common block
    COMMON foxsi_smex_vars, foxsi_root_path, foxsi_data_path, foxsi_name, $
    foxsi_optic_effarea, foxsi_number_of_modules, foxsi_shell_ids, $
    foxsi_shutters_thickness_mm, foxsi_detector_thickness_mm, foxsi_blanket_thickness_mm, $
    foxsi_stc_aperture_area_mm2, foxsi_stc_detector_material, foxsi_stc_detector_thickness_mm, $
    foxsi_stc_filter_material, foxsi_stc_filter_thickness_um

    data_filename = 'effective_area_per_module.csv'
    path = foxsi_data_path + data_filename
    ;path = '../../data/' + data_filename

    number_of_columns = 2
    number_header_lines = 5
    number_lines = FILE_LINES(path)
    number_data_lines = number_lines - number_header_lines; - 10

    OPENR, lun, path, /GET_LUN
    header = strarr(number_header_lines)
    readf, lun, header
    data = fltarr(number_of_columns, number_data_lines)
    readf, lun, data
    free_lun, lun
    result = create_struct('energy_keV', data[0, *], 'eff_area_cm2_1', data[1, *])

    RETURN, result
END
