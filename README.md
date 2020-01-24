# Simple Thai OCR
Simple Thai Optical Character Recognition<br>
[Read more in Thai](https://medium.com/@p.siriphanthong/thai-optical-character-recognition-thai-ocr-%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B9%81%E0%B8%9B%E0%B8%A5%E0%B8%87%E0%B8%A3%E0%B8%B9%E0%B8%9B%E0%B8%A0%E0%B8%B2%E0%B8%9E%E0%B9%80%E0%B8%9B%E0%B9%87%E0%B8%99%E0%B8%82%E0%B9%89%E0%B8%AD%E0%B8%84%E0%B8%A7%E0%B8%B2%E0%B8%A1-fdeede331b6d)

### Overview
1. Row Segmentation - Use Horizontal Projections and Combine Rows Algorithms
2. Character Segmentation - Use Blob Coloring Algorithm
3. Character Sorting - Use Consonant Line Algorithm
4. Character Recognition - Use Template Matching and Sorting Alphabets Algorithms

### Errors
|            | Error Segmentation | Error Recognition | Error Spacing |
|:---|:---:|:---:|:---:|
| Same Font, Same Size, No Noise | 2.09% | 0.00% | 6.1% |
| Same Font, Different Size, No Noise | 1.05% | 0.00% | 3.0% |
| Same Font, Different Size, Scanned | 2.09% | 0.75% | 6.1% |
| Different Font, No Noise | 1.05% | 63.60% | 39.4% |
| Different Font, Scanned | 2.09% | 71.32% | 97.0% |

_** The font family of templates is 'Angsana New' and size is 12px_
