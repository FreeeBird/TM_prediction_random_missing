'''
Author: FreeeBird
Date: 2022-04-14 11:57:56
LastEditTime: 2022-04-14 15:47:55
LastEditors: FreeeBird
Description: 
FilePath: /TM_Prediction_With_Missing_Data/utils/excel_helper.py
'''
import openpyxl
 
 
def write_excel_xlsx(path, values):
    # index = len(value)
    workbook = openpyxl.load_workbook(path)
    sheet = workbook.active
    for line in values:
            sheet.append(line)
    # sheet.title = sheet_name
    workbook.save(path)
 
 
def read_excel_xlsx(path, sheet_name):
    workbook = openpyxl.load_workbook(path)
    sheet = workbook[sheet_name]
    for row in sheet.rows:
        for cell in row:
            print(cell.value, "\t", end="")
        print()
 
 
book_name_xlsx = '/home/liyiyong/TM_Prediction_With_Missing_Data/result.xlsx'
 
sheet_name_xlsx = 'Sheet1'
 
value3 = [
    ["0", "0", "0", "0", "0","0", "0", "0", "0", "0","0", "0", "0", "0", "0","0", "0", "0"],
    ["0", "0", "0", "0", "0","0", "0", "0", "0", "0","0", "0", "0", "0", "0","0", "0", "0"],
    ["0", "0", "0", "0", "0","0", "0", "0", "0", "0","0", "0", "0", "0", "0","0", "0", "0"],
    ]
 
 
if __name__ == '__main__':
    read_excel_xlsx(book_name_xlsx, sheet_name_xlsx)
    # write_excel_xlsx(book_name_xlsx,sheet_name_xlsx, value3)
    # read_excel_xlsx(book_name_xlsx, sheet_name_xlsx)
