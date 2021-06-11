

def read_to_csv(file_):
    line_latex = [
 
        '\\begin{table}',
        '\\noindent|\makebox[\\textwidth]{%',
        '\\begin{tabular}{| c | c | c | c | c | c | c | c | c | c | c | c | c |}'
    ]


    with open(file_, 'r') as reader:
        for line in reader.readlines():
            line_latex.append('&'.join(line.split(',')[:-1])  + ' \\\\')
   

    line_latex[3] = line_latex[3] + ' \\hline' 
    line_latex[-1] = line_latex[-1] + ' \\hline'  


    line_latex += [

        '\\end{tabular}}',
        '\\caption{Fruta disponible}',
        '\\label{tab:fruta}',
        '\\end{table} '
    ]


    return line_latex

  
def save_to_latex(file_, line_latex):
    
    f = open(file_, 'w')

    for line in line_latex:
	line = '-'.join(line.split('_'))
        f.write(line)
	f.write('\n')
    f.close()



if __name__ == '__main__':
    

    model = 'faster_rcnn_R_50_FPN_3xmetric_image.csv'
    line_latex = read_to_csv(model)    
    save_to_latex(model + '.latex', line_latex)


    model = 'faster_rcnn_R_101_FPN_3xmetric_image.csv'
    line_latex = read_to_csv(model)    
    save_to_latex(model + '.latex', line_latex)


    model = 'mask_rcnn_R_50_FPN_3xmetric_image.csv'
    line_latex = read_to_csv(model)    
    save_to_latex(model + '.latex', line_latex)



    model = 'mask_rcnn_R_101_FPN_3xmetric_image.csv'
    line_latex = read_to_csv(model)    
    save_to_latex(model + '.latex', line_latex)


    model = 'mask_rcnn_X_101_32x8d_FPN_3xmetric_image.csv'
    line_latex = read_to_csv(model)    
    save_to_latex(model + '.latex', line_latex)


    model = 'faster_rcnn_X_101_32x8d_FPN_3xmetric_image.csv'
    line_latex = read_to_csv(model)    
    save_to_latex(model + '.latex', line_latex)












