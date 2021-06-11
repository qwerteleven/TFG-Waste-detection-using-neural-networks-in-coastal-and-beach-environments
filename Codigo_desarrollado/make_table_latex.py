

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
        f.write(line)
	f.write('\n')
    f.close()



if __name__ == '__main__':
    
    import glob
    models = glob.glob("./*.csv")
    for model in models:
        line_latex = read_to_csv(model)    
        save_to_latex("./latex-tables" + model + '.latex', line_latex)












