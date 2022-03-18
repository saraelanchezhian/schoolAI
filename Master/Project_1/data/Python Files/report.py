import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
import base64
import pdfkit

def report(feat, daily_outputs):
    
    """"""
    features = feat
    dout = daily_outputs
    dout_Y3 = dout[dout["Y_PRED"]==3]
    dout_3 = len(dout[dout["Y_PRED"]==3])
    dout_2 = len(dout[dout["Y_PRED"]==2])
    dout_1 = len(dout[dout["Y_PRED"]==1])
    dout_0 = len(dout[dout["Y_PRED"]==0])
    
    pivot = pd.pivot_table(dout_Y3,  
                           index=['Case.ID'],
                           values=['X1','X2','X3','X4','X5','X6','X7','X8','X9',
                                   'X10','X11','X12','X13','X14','X15','X16','X17',
                                   'X18','X19','X20','X21','X22','X23','X24','X25',
                                   'X26','X27','X28']
                           )
    pivot.columns = features['Feature']

    """"""
    data = [dout_3, dout_2, dout_1, dout_0]
    flag = ['0','1','2','3']
    plt.pie(data, labels = flag, shadow=True)
    plt.legend(plt , labels = flag,
              loc ="upper right", bbox_to_anchor=(1.2, 1))
    plt.figtext(1.5,0.8,'Summary', fontsize=20, wrap=True, fontweight='medium', bbox={'facecolor':'grey','alpha':0.3, 'pad':5})
    plt.figtext(1.4,0.65,'Number of Level 3 Cases: ', fontsize=14, color='red')
    plt.figtext(1.85,0.65,dout_3, fontsize=14, color='red')
    plt.figtext(1.4,0.55,'Number of Level 2 Cases: ', fontsize=14, color='green')
    plt.figtext(1.85,0.55,dout_2, fontsize=14, color='green')
    plt.figtext(1.4,0.45,'Number of Level 1 Cases: ', fontsize=14, color='orange')
    plt.figtext(1.85,0.45,dout_1, fontsize=14, color='orange')
    plt.figtext(1.4,0.35,'Number of Level 0 Cases: ', fontsize=14, color='blue')
    plt.figtext(1.85,0.35,dout_0, fontsize=14, color='blue')
    plt.savefig('C:/Users/gayat/Documents/AI Sara/Master/Project_1/data/piechart.png', bbox_inches = "tight", dpi=150)

    
    """"""
    data_uri = base64.b64encode(open('C:/Users/gayat/Documents/AI Sara/Master/Project_1/data/piechart.png', 'rb').read()).decode('utf-8')
    image_tag = '<center><img src="data:image/png;base64,{0}"></center>'.format(data_uri)
    
    #writing HTML Content
    
    heading = '<center><h1 style="color:red;font-size:60px;"> Threat level Assessment Report </h1></center>'
    subheading = '<center><h3 style="color:blue;"> [Sample Report] </h3></center>'
    # Using .now() from datetime library to add Time stamp
    now = datetime.now()
    current_time = now.strftime("%m/%d/%Y %H:%M:%S")
    header = '<div class="top">' + heading + subheading +'</div>'
    footer = '<div class="bottom"> <h3> This Report has been Generated\n on '+ current_time +'</h3> </div>'
    content = '<div class="chart"> '+image_tag+'</div> \n <div class="table"> '+pivot.to_html()+' </div>'
    # Concating everything to a single string
    html = header + content + footer

    # Writing the file
    with open('C:/Users/gayat/Documents/AI Sara/Master/Project_1/data/report.html','w+') as file:
     file.write(html)
    
    """"""
    config = pdfkit.configuration(wkhtmltopdf='C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe')
    pdfkit.from_file('C:/Users/gayat/Documents/AI Sara/Master/Project_1/data/report.html',
                     'C:/Users/gayat/Documents/AI Sara/Master/Project_1/data/report.pdf', 
                     configuration=config, options={'orientation':'Landscape'})
     