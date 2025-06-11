import os
import json

JSON_PATH = '/Volumes/Storage/JSONs_old_new/thelast_00019002a_mo4724_pS1.96536799834280303030.json'

with open(JSON_PATH, 'r') as file:
    json_dict = json.load(file)

pred = json_dict['00019002a_mo4724_pS1.96536799834280303030.tif']['predictions']
ring_line_n = len(pred['ring_line'])
ring_poly_n = len(pred['ring_polygon'])
crack_poly_n = len(pred['crack_polygon'])
ring_line_lengths = [len(ring_d['x']) for _, ring_d in iter(pred['ring_line'].items())]
ring_lines_cum_points = sum(ring_line_lengths)

ring_poly_lengths = [len(ring_d['x']) for _, ring_d in iter(pred['ring_polygon'].items())]
ring_poly_cum_points = sum(ring_poly_lengths)

crack_poly_lengths = [len(crack_d['x']) for _, crack_d in iter(pred['crack_polygon'].items())]
crack_poly_cum_points = sum(crack_poly_lengths)

OUT_PATH = '/Volumes/Storage/JSONs_old_new/thelast_00019002a_mo4724_pS1.96536799834280303030.txt'
with open(OUT_PATH, 'w') as f:
    f.write(f'Ring lines: {ring_line_n} \n'
            f'Ring polygons: {ring_poly_n} \n'
            f'Crack polygons: {crack_poly_n} \n'
            f'Ring line points: {ring_lines_cum_points} \n'
            f'Ring polygon points: {ring_poly_cum_points} \n'
            f'Crack polygon points: {crack_poly_cum_points} \n')