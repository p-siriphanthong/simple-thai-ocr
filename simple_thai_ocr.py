import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

figsize = (20, 10)
src_path = 'input.jpg'
output_filename = 'output.txt'
template_path = 'Templates'

def ConvertToGray(src):
    R = src[:, :, 0]
    G = src[:, :, 1]
    B = src[:, :, 2]
    return np.round(0.2989 * R + 0.5870 * G + 0.1140 * B).astype(int)

def ConvertToBW(src):
    bw = np.copy(src)
    threshold = AutomaticThresholdBW(src)
    bw[np.where(bw <= threshold)] = 0
    bw[np.where(bw > threshold)] = 255
    return bw
    
def AutomaticThresholdBW(src):
    T = np.mean(src)
    while True:
        R1 = src[np.where(src <= T)]
        R2 = src[np.where(src > T)]
        new_T = (np.mean(R1) + np.mean(R2)) / 2
        if T == new_T:
            return T
        T = new_T

def OCR(src, filename):
    file = open(filename, 'w')
    gray = ConvertToGray(src)
    rows = RowSegmentation(gray)
    templates = GetTemplates()
    for row in rows:
        bounding_boxes, consonant_line, space_threshold = BlobColoring(gray[row['start'] : row['stop']])
        sorted_alphabets = SortBoundingBox(bounding_boxes, consonant_line)
        
        index = 0
        have_message = False
        while index < len(sorted_alphabets):
            box = sorted_alphabets[index]
            cv2.rectangle(src[row['start'] : row['stop']], box['top_left'] ,box['bottom_right'], (31, 119, 180), 2)
            
            if box['position'] == 'middle':
                if 'consonant_before' in locals() and box['top_left'][0] - consonant_before['bottom_right'][0] >= space_threshold:
                    print(' ', end = '')
                    file.write(' ')
                consonant_before = box
            
            alphabet = TemplateMatching(box, templates)
            have_message = have_message or alphabet != ''
            print(alphabet, end = '')
            file.write(alphabet)
            
            if alphabet in ['i', 'ำ', 'ะ']:
                index += 1
                if index < len(sorted_alphabets) and sorted_alphabets[index]['position'] == 'middle':
                    consonant_before = sorted_alphabets[index]
            index += 1
        
        if have_message:
            file.write('\n')        
            
        src[row['start'] + consonant_line - 1 : row['start'] + consonant_line + 2, : ] = [255, 0, 0]
        plt.figure(figsize = figsize, dpi = 100)
        plt.imshow(src[row['start'] : row['stop']])
        plt.xticks([]), plt.yticks([])
        plt.show()
        
    plt.figure(figsize = figsize, dpi = 100)
    plt.imshow(src)
    plt.xticks([]), plt.yticks([])
    plt.show()
    
    file.close()

def RowSegmentation(src, threshold = 0.2):
    src = ConvertToBW(src)
    horizontal_projections = np.zeros(src.shape[0], dtype = int)
    previous = 0
    row_lists = []
    for row in range(src.shape[0]):
        horizontal_projections[row] = len(np.where(src[row] == 0)[0])
        if horizontal_projections[row] != 0 and previous == 0:
            row_lists.append({'start' : row})
        elif horizontal_projections[row] == 0 and previous != 0:
            row_lists[-1]['stop'] = row
        previous = horizontal_projections[row]
    
    plt.figure(figsize = figsize)
    plt.plot(horizontal_projections, np.arange(src.shape[0]))
    plt.imshow(src, 'gray')
    plt.xticks([]), plt.yticks([])
    plt.show()
    
    mark = np.empty((src.shape[0], src.shape[1], 3), dtype = 'uint8')
    mark.fill(255)
    for i, row in enumerate(row_lists):
        if i % 2 == 0:
            mark[row['start'] : row['stop'], :] = [31, 119, 180]
        else:
            mark[row['start'] : row['stop'], :] = [255, 127, 14]
            
    plt.figure(figsize = figsize)
    plt.imshow(src, 'gray')
    plt.imshow(mark, alpha = 0.15)
    plt.xticks([]), plt.yticks([])
    plt.show()
    
    i = 0
    while i < len(row_lists) - 1:
        if i == 0:
            space_top = row_lists[i]['start']
        else:
            space_top = row_lists[i]['start'] - row_lists[i-1]['stop']
        space_bottom = row_lists[i+1]['start'] - row_lists[i]['stop']
        if space_bottom / space_top <= threshold:
            row_lists[i]['stop'] = row_lists[i+1]['stop']
            row_lists.pop(i+1)
        else:
            i += 1
    
    mark = np.empty((src.shape[0], src.shape[1], 3), dtype = 'uint8')
    mark.fill(255)
    for i, row in enumerate(row_lists):
        if i % 2 == 0:
            mark[row['start'] : row['stop'], :] = [31, 119, 180]
        else:
            mark[row['start'] : row['stop'], :] = [255, 127, 14]
            
    plt.figure(figsize = figsize)
    plt.imshow(src, 'gray')
    plt.imshow(mark, alpha = 0.15)
    plt.xticks([]), plt.yticks([])
    plt.show()

    return row_lists

def BlobColoring(src):
    bw = ConvertToBW(src)
    label = -1
    equivalence_labels = []
    
    for i in range(bw.shape[0]):
        for j in range(bw.shape[1]):
            if bw[i][j] == 0:
                neighbors = []
                neighbors.extend([bw[i-1][j-1], bw[i-1][j], bw[i-1][j+1], bw[i][j-1]])
                neighbors = list(set(neighbors))
                if 255 in neighbors:
                    neighbors.remove(255)
                if len(neighbors) == 0:
                    bw[i][j] = label
                    equivalence_labels.append(set([label]))
                    label -= 1
                else:
                    bw[i][j] = max(neighbors)
                    equivalence_labels.append(set(neighbors))
    
    i = 0
    while i < len(equivalence_labels):
        j = i + 1
        while j < len(equivalence_labels):
            if equivalence_labels[i].intersection(equivalence_labels[j]):
                equivalence_labels[i] = equivalence_labels[i].union(equivalence_labels.pop(j))
                j = i + 1
            else:
                j += 1
        i += 1
    
    bounding_boxes = []
    consonant_line = 0
    space_threshold = 0
    for index, equivalence_label in enumerate(equivalence_labels, 1):
        for row in range(bw.shape[0]):
            for col in range(bw.shape[1]):
                if bw[row][col] in equivalence_label:
                    bw[row][col] = index

        box = {
                'id' : index,
                'src' : src[min(np.where(bw == index)[0]) : max(np.where(bw == index)[0]) + 1, 
                            min(np.where(bw == index)[1]) : max(np.where(bw == index)[1]) + 1],
                'position_x' : (max(np.where(bw == index)[1]) - min(np.where(bw == index)[1]) + 1) / 2 + min(np.where(bw == index)[1]),
                'position_y' : (max(np.where(bw == index)[0]) - min(np.where(bw == index)[0]) + 1) / 2 + min(np.where(bw == index)[0]),
                'top_left' : tuple([min(np.where(bw == index)[1]), min(np.where(bw == index)[0])]),
                'bottom_right' : tuple([max(np.where(bw == index)[1]), max(np.where(bw == index)[0])])
              }
        bounding_boxes.append(box)
        consonant_line += box['position_y']
        space_threshold += box['src'].shape[1]
        
    consonant_line = round(consonant_line / len(equivalence_labels) + 0.05 * bw.shape[0]).astype(int)
    space_threshold = (space_threshold / len(equivalence_labels)) * 0.45
    
    return bounding_boxes, consonant_line, space_threshold

def GetTemplates():
    templates = []
    for path, subdirs, files in os.walk(template_path):
        for name in files:
            file_path = os.path.join(path, name)
            template = ConvertToGray(mpimg.imread(file_path, format = 'jpg'))
            obj = {}
            obj['character'] = os.path.split(file_path)[-1].split('.')[0]
            obj['position'] = os.path.split(file_path)[-1].split('.')[1]
            obj['amount_top'] = [int(amount) for amount in os.path.split(file_path)[-1].split('.')[2].split(',')]
            obj['amount_bottom'] = [int(amount) for amount in os.path.split(file_path)[-1].split('.')[3].split(',')]
            obj['src'] = template
            obj['ratio'] = template.shape[1] / template.shape[0]
            templates.append(obj)
    return templates

def TemplateMatching(box, templates, ratio_threshold = 0.2, size_threshold = 30):
    if box['src'].size >= size_threshold:
        src_ratio = box['src'].shape[1] / box['src'].shape[0]
        templates_filtered = [template for template in templates if abs(template['ratio'] - src_ratio) <= ratio_threshold]
        templates_filtered = [template for template in templates_filtered if template['position'] == box['position']]
        templates_filtered = [template for template in templates_filtered if box['amount_top'] in template['amount_top']]
        templates_filtered = [template for template in templates_filtered if box['amount_bottom'] in template['amount_bottom']]
        for template in templates_filtered:
            reduce_src = ReduceSize(box['src'], template['src'].shape[0], template['src'].shape[1])
            error = np.sum(np.abs(reduce_src - template['src'])) / reduce_src.size
            if 'match' not in locals() or error < match['error']:
                match = { 'error' : error, 'character' : template['character'] }
        if 'match' in locals():
            return match['character']
    return ''

def ReduceSize(src, height, width):
    output = np.empty((height, width), dtype = 'uint8')
    h_ratio = src.shape[0] / height
    w_ratio = src.shape[1] / width
    for i in range(height):
        for j in range(width):
            start_row = [int(i * h_ratio), 1 - (i * h_ratio) % 1]
            end_row = [int((i + 1) * h_ratio), ((i + 1) * h_ratio) % 1]
            start_col = [int(j * w_ratio), 1 - (j * w_ratio) % 1]
            end_col = [int((j + 1) * w_ratio), ((j + 1) * w_ratio) % 1]
            weight = np.ones((end_row[0] - start_row[0] + 1, end_col[0] - start_col[0] + 1))
            weight[0, :] *= start_row[1]
            weight[-1, :] *= end_row[1]
            weight[:, 0] *= start_col[1]
            weight[:, -1] *= end_col[1]
            if end_row[0] >= src.shape[0]:
                weight = np.delete(weight, -1, axis = 0)
            if end_col[0] >= src.shape[1]:
                weight = np.delete(weight, -1, axis = 1)
            sub_src = src[start_row[0] : start_row[0] + weight.shape[0], start_col[0] : start_col[0] + weight.shape[1]]
            output[i][j] = round(np.sum(sub_src * weight) / np.sum(weight))
    return output

def SortBoundingBox(bounding_boxes, consonant_line):
    sorted_alphabets = []
    
    consonants = [box for box in bounding_boxes if consonant_line >= box['top_left'][1] and consonant_line <= box['bottom_right'][1]]
    consonants = sorted(consonants, key = lambda consonant: consonant['position_x'])
    
    for consonant in consonants:
        consonant['position'] = 'middle'
        top_bottom = [box for box in bounding_boxes if box['position_x'] >= consonant['top_left'][0] and box['position_x'] <= consonant['bottom_right'][0]]
        tops = [box for box in top_bottom if box['position_y'] < consonant['position_y']]
        bottoms = [box for box in top_bottom if box['position_y'] > consonant['position_y']]
        
        consonant['amount_top'] = len(tops)
        consonant['amount_bottom'] = len(bottoms)
        sorted_alphabets.append(consonant)
        
        if len(bottoms):
            bottoms[0]['position'] = 'bottom'
            bottoms[0]['amount_top'] = 1 + len(tops)
            bottoms[0]['amount_bottom'] = 0
            sorted_alphabets.append(bottoms[0])
        
        if len(tops):
            tops = sorted(tops, key = lambda box: box['position_y'], reverse = True)
            for index, top in enumerate(tops):
                top['position'] = 'top'
                top['amount_top'] = len(tops) - index - 1
                top['amount_bottom'] = index + 1 + len(bottoms)
            sorted_alphabets.extend(tops)

    for box in bounding_boxes:
        if len([sorted_alphabet for sorted_alphabet in sorted_alphabets if sorted_alphabet['id'] == box['id']]) == 0:
            box['position'] = 'middle'
            box['amount_top'] = 0
            box['amount_bottom'] = 0
            index = [index for index, sorted_alphabet in enumerate(sorted_alphabets) if sorted_alphabet['position_x'] > box['position_x']]
            if len(index):
                index = min(index)
                sorted_alphabets.insert(index, box)
            else:
                sorted_alphabets.append(box)

    return sorted_alphabets

def main():
    img = mpimg.imread(src_path, format = 'jpg')
    OCR(img, output_filename)

if __name__ == '__main__':
    main()