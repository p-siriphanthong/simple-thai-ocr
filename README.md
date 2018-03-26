# Simple Thai OCR
Simple Thai Optical Character Recognition

### Overview
1. Segmentation of Rows - Use Horizontal Projections and Combine Rows Algorithms
2. Segmentation of Characters - Use Blob Coloring Algorithm
3. Character Recognition - Use Template Matching, Consonant Line and Sorting Alphabets Algorithms

### Errors
|            | Error Segmentation | Error Recognition | Error Spacing |
|:---|:---:|:---:|:---:|
| Same Font, Same Size, No Noise | 2.09% | 0.00% | 6.1% |
| Same Font, Different Size, No Noise | 1.05% | 0.00% | 3.0% |
| Same Font, Different Size, Scanned | 2.09% | 0.75% | 6.1% |
| Different Font, No Noise | 1.05% | 63.60% | 39.4% |
| Different Font, Scanned | 2.09% | 71.32% | 97.0% |

_** The font family of templates is 'Angsana New' and size is 12px_
