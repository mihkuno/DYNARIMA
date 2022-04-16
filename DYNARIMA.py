from PyQt5 import QtCore, QtGui, uic
from PyQt5.QtWidgets import (
    QDialog, 
    QApplication, 
    QMainWindow, 
    QPushButton, 
    QVBoxLayout, 
    QCalendarWidget, 
    QSpinBox, 
    QLabel, 
    QDateEdit, 
    QWidget, 
    QProgressBar)

import pandas as pd
import random
import os   
import sys

# ---- INITIALIZE DIRECTORIES ----
dir_dataset = 'dataset/'
dir_config = 'config/'
dir_current = os.getcwd() # get the current current location

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

# ---- INITIALIZE MATPLOTLIB ----
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib import pyplot as plt, dates as mdates
        
class DYNARIMA(QMainWindow):
    def __init__(self):
        super(DYNARIMA, self).__init__()
        self.ui = uic.loadUi(resource_path('DESIGN.ui'), self) # load ui file
        self.prev_lags = self.new_lags = 1 # config_lags highlighter
        self.dataframe = self.trainset = self.testset = ''
        
        # ---- ASSOCIATE ELEMENTS TO VARIABLE ----
        # buttons
        self.btn_compile = self.findChild(QPushButton, 'btn_compile')
        self.btn_forecast = self.findChild(QPushButton, 'btn_forecast')
        self.btn_webscrape = self.findChild(QPushButton, 'btn_webscrape')
        
        self.btn_compile.clicked.connect(self.compile)
        self.btn_forecast.clicked.connect(self.forecast)
        self.btn_webscrape.clicked.connect(self.webscrape)
        
        # configs
        self.calendar = self.findChild(QCalendarWidget, 'calendar')
        self.calendar.selectionChanged.connect(self.calendar_)
        self.config_integrate = self.findChild(QSpinBox, 'config_integrate')
        self.config_lags = self.findChild(QSpinBox, 'config_lags')
        self.config_startdate = self.findChild(QDateEdit, 'config_startdate')
        
        # setup startdate
        self.selected_date = QtCore.QDate.currentDate()
        self.config_startdate.setDate(self.selected_date)
        self.calendar.setSelectedDate(self.selected_date)
        self.config_lags.valueChanged.connect(self.lags)
        self.config_startdate.dateChanged.connect(self.startdate)
        self.config_integrate.valueChanged.connect(self.integrate)
        
        # informations
        self.txt_accuracy = self.findChild(QLabel, 'info_txt_accuracy')
        self.txt_adf = self.findChild(QLabel, 'info_txt_adf')
        self.txt_aic = self.findChild(QLabel, 'info_txt_aic')
        self.txt_mae = self.findChild(QLabel, 'info_txt_mae')
        self.txt_model = self.findChild(QLabel, 'info_txt_model')
        self.txt_pvalue = self.findChild(QLabel, 'info_txt_pvalue')
        
        # matplot qwidgets
        self.matplot_container = self.findChild(QVBoxLayout, 'matplot_container')
                
        # progress bars
        self.progressbar = self.findChild(QProgressBar, 'progressbar')
        self.progressbar_text = self.findChild(QLabel, 'progressbar_text')       
        print('-- LINKED UI AND LOGIC --')
        
        # ---- INITIALLY LOCK AND SET TEXT ----
        # index -1 initial (lock all except webscraper/update button)
        # index 0 webscrape 
        # index 1 compile
        # index 2 forecast
        self.locker(-1)
        print('-- LOCKED SEQUENCE --')
        
        
        # ---- CREATE PREREQUISITE FILES ----
        # if not found, create dataset and  folder with configurations 
        self.toScrape = True
        if not os.path.exists(dir_dataset):
                os.makedirs(dir_dataset)
        elif len(os.listdir(dir_dataset)) >= 4:
            self.btn_webscrape.setText('Compress')
            self.toScrape = False
        if not os.path.exists(dir_config):
            os.makedirs(dir_config)
        if len(os.listdir(dir_config)) < 2:
            import json
            # Google API OAuth2.0
            client_secrets = json.dumps(
                {"web":{
                    "client_id":"626625711266-90bhqs8j4vj9cru2jre94cbqamn7e9j8.apps.googleusercontent.com",
                    "project_id":"original-bot-295405",
                    "auth_uri":"https://accounts.google.com/o/oauth2/auth",
                    "token_uri":"https://oauth2.googleapis.com/token","auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs","client_secret":"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",
                    "redirect_uris":["http://localhost:8080/"],
                    "javascript_origins":["http://localhost:8080"]}
                })
            # Matplotlib Stylesheet
            matplotstyle = """
                # Seaborn common parameters
                # .15 = dark_gray
                # .8 = light_gray
                legend.frameon: False
                legend.numpoints: 1
                legend.scatterpoints: 1
                xtick.direction: out
                ytick.direction: out
                axes.axisbelow: True
                font.family: sans-serif
                grid.linestyle: -
                lines.solid_capstyle: round

                # Seaborn darkgrid parameters
                axes.grid: True
                axes.edgecolor: white
                axes.linewidth: 0
                xtick.major.size: 0
                ytick.major.size: 0
                xtick.minor.size: 0
                ytick.minor.size: 0

                # from mplcyberpunk
                text.color: 0.9
                axes.labelcolor: 0.9
                xtick.color: 0.9
                ytick.color: 0.9
                grid.color: 2A3459

                # Custom
                font.sans-serif: Overpass, Helvetica, Helvetica Neue, Arial, Liberation Sans, DejaVu Sans, Bitstream Vera Sans, sans-serif
                axes.prop_cycle: cycler('color', ['18c0c4', 'f62196', 'A267F5', 'f3907e', 'ffe46b', 'fefeff'])
                image.cmap: RdPu
                figure.facecolor: 212946
                axes.facecolor: 212946
                savefig.facecolor: 212946
                """
            # write the files to the config
            with open(f"{dir_config}client_secrets.json", "w") as outfile:
                outfile.write(client_secrets)
            with open(f"{dir_config}matplotlib-dark.mplstyle", "w") as outfile:
                outfile.write(matplotstyle)
        plt.style.use(dir_config+'matplotlib-dark.mplstyle') # change the matplotlib theme
        print('-- PREREQUISITES COMPLETE --')    
        
    def compile(self):
        if self.btn_compile.text() == 'Compile':
            self.progressbar_text.setText('Compiling...')
            self.progressbar.setValue(0)
            start = self.config_startdate.date().toString("yyyy-MM-dd") # starting date
            predict = self.config_lags.value() # number of days to forecast
            
            print('-- SPLITTING TRAIN AND TEST DATASETS --')
            # train date
            train_fr = pd.to_datetime('Jan 30, 2020').date() # DO NOT TOUCH
            train_to = (pd.to_datetime(start) + pd.DateOffset(days=-1)).date()
            # test date
            test_fr  =  pd.to_datetime(start).date()
            test_to  =  (test_fr + pd.DateOffset(days=predict-1)).date()
            # split dates of the dataset
            self.trainset = self.dataframe.loc[train_fr:train_to].rename(columns={'Cases':'Train'})
            self.testset  = self.dataframe.loc[test_fr:test_to].rename(columns={'Cases':'Test'})
            self.testset.Cases = ['Test']
            
            print('--- TRAINING DATASET ---')
            print(self.trainset)
            print('--- TESTING DATASET ---')
            print(self.testset)
            
            title = f'Train-Test Split [{self.trainset.index[0]} — {self.testset.index[-1]}]'
            self.clearLayout(self.matplot_container)
            df_split = Plotter(title, self.trainset, self.testset)
            self.matplot_container.addWidget(df_split)
            
            self.progressbar_text.setText('Compilation Complete')
            self.progressbar.setValue(100)
            self.locker(1)
        elif self.btn_compile.text() == 'Reset':
            self.progressbar.setValue(0)
            self.locker(0)
            self.progressbar_text.setText('Dataframe Reset')
    
    def webscrape(self):
        self.locker() # lock eveything when this button is clicked
        if self.toScrape or self.btn_webscrape.text() == 'Update':
            self.thread = ThreadClass(parent=None, index=0, toScrape=True)
        else:
            self.thread = ThreadClass(parent=None, index=0, toScrape=False)
        # start and configure locker when this thread is done
        self.thread.progress_signal.connect(self.progress_worker)
        self.thread.output_signal.connect(self.output_worker)
        self.thread.start() 
             
    def forecast(self):
        self.locker()
        self.thread = ThreadClass(parent=None, index=2, dataframe=[self.trainset, self.testset, self.config_integrate.value()])
        self.thread.progress_signal.connect(self.progress_worker)
        self.thread.output_signal.connect(self.output_worker)
        self.thread.start() 
        
    def lags(self):
        format, date, init = self.reset_highlight()
        self.new_lags = self.config_lags.value()
        # reset initial selected position
        self.calendar.setSelectedDate(date)
        self.update_highlight()
        self.prev_lags = self.new_lags
        
    def startdate(self):
        # updates the calendar
        self.reset_highlight()
        self.calendar.setSelectedDate(self.config_startdate.date())
        self.selected_date = self.config_startdate.date()
        self.update_highlight()
        
    def integrate(self):
        diff = self.config_integrate.value()
        self.txt_model.setText(f"Model: {(7, diff, 8)}")
        
    def calendar_(self):
        # updates the startdate config
        self.reset_highlight()
        self.config_startdate.setDate(self.calendar.selectedDate())
        self.selected_date = self.config_startdate.date()
        self.update_highlight()
    
    def reset_highlight(self):
        # reset highlights
        format = QtGui.QTextCharFormat()  
        date = self.selected_date
        init = self.calendar.palette().brush(QtGui.QPalette.Base)
        format.setBackground(init)
        for i in range(self.prev_lags):
            self.calendar.setDateTextFormat(date.addDays(i), format)
        return format, date, init
    
    def update_highlight(self):
        # change the temporary up-date to the number of lags  
        format = QtGui.QTextCharFormat()  
        date = self.selected_date
        active = self.calendar.palette().brush(QtGui.QPalette.LinkVisited)
        format.setBackground(active)
        for i in range(self.new_lags):
            self.calendar.setDateTextFormat(date.addDays(i), format)
    
    def locker(self, index=None):
        # lock everything
        self.btn_compile.setText('Compile')
        self.calendar.setDisabled(True)
        self.btn_compile.setDisabled(True)
        self.btn_forecast.setDisabled(True)
        self.btn_webscrape.setDisabled(True)
        self.config_integrate.setDisabled(True)
        self.config_lags.setDisabled(True)
        self.config_startdate.setDisabled(True)
        
        self.txt_adf.setText('ADF: 0.0')
        self.txt_pvalue.setText('P-Value: 0.0')
        self.txt_aic.setText('AIC: 0.0')
        self.txt_model.setText(f'Model: {(7, self.config_integrate.value(), 8)}' )
        self.txt_mae.setText('MAE: 0%')
        self.txt_accuracy.setText('Accuracy: 0%')
        
        if index==-1: # initial webscrape lock
            self.btn_webscrape.setText('Webscrape')
            self.btn_webscrape.setDisabled(False)
        elif index==0 or index==2: # webscrape / forecast finished, what to unlock?
            self.btn_webscrape.setText('Update')
            self.calendar.setDisabled(False)
            self.btn_webscrape.setDisabled(False)
            self.btn_compile.setDisabled(False)
            self.config_lags.setDisabled(False)
            self.config_startdate.setDisabled(False)
        elif index==1: # compile finished, what to unlock?
            self.btn_webscrape.setText('Update')
            self.btn_compile.setText('Reset')
            self.btn_webscrape.setDisabled(False)
            self.btn_compile.setDisabled(False)
            self.btn_forecast.setDisabled(False)
            self.config_integrate.setDisabled(False)

    # update the progressbar from thread emits
    def progress_worker(self, counter, title, activate):
        index = self.sender().index
        self.progressbar.setValue(counter)
        self.progressbar_text.setText(title)
        if activate: # activates locker based on index
            self.locker(index)
    
    # get the dataframe from the worker thread
    def output_worker(self, data):
        index = self.sender().index
        if index==0: # get output of webscrape
            self.dataframe = data[0]
            self.calendar.setMaximumDate(data[0].index[-1])
            self.config_startdate.setMaximumDate(data[0].index[-1])
            
            title = f'Webscraped Dataframe [{data[0].index[0]} — {data[0].index[-1]}]'
            df_canvas = Plotter(title, data[0]) 
            self.clearLayout(self.matplot_container)
            self.matplot_container.addWidget(df_canvas) # adding canvas to the layout
            
            print('-- CANVAS PLOTTER CREATED --')
        elif index==2: # get output of forecast
            adf, pvalue, aic, model, mae, accuracy, df_test, fit_model = data    
            print(adf, pvalue, aic, model, mae, accuracy)

            self.txt_adf.setText(f'ADF: {round(adf,8)}')
            self.txt_pvalue.setText(f'P-Value: {round(pvalue,8)}')
            self.txt_aic.setText(f'AIC: {round(aic,8)}')
            self.txt_model.setText(f'Model: {model}')
            self.txt_mae.setText(f'MAE: {round(mae,2)}%')
            self.txt_accuracy.setText(f'Accuracy: {round(accuracy,2)}%')    
            
            plt.rc('font', size=6) # controls default text sizes
            plt.tight_layout()

            model_diagnostic = FigureCanvas(fit_model.plot_diagnostics(figsize=(30,10)))
            model_comparison = Plotter(f'Accuracy Analysis', df_test['Test'], df_test['Model'])
            self.clearLayout(self.matplot_container)
            self.matplot_container.addWidget(model_comparison) # adding canvas to the layout
            self.matplot_container.addWidget(model_diagnostic)   
               
    def clearLayout(self, layout):
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget() is not None:
                    child.widget().deleteLater()
                elif child.layout() is not None:
                    self.clearLayout(child.layout())

class ThreadClass(QtCore.QThread): 
    progress_signal = QtCore.pyqtSignal(int, str, bool)
    output_signal = QtCore.pyqtSignal(list)
    def __init__(self, parent=None, index=0, toScrape=False, dataframe=[]):
        super(ThreadClass, self).__init__(parent)
        self.index = index
        self.is_running = True
        self.toScrape = toScrape
        self.dataframe = dataframe
        
    def run(self):
        if self.index==0: # webscrape
            if self.toScrape: # if dataset is missing
                # --AUTHENTICATE--
                from pydrive.auth import GoogleAuth
                from pydrive.drive import GoogleDrive
                # reset progress bar
                self.progress_signal.emit(0,'Authenticate Google Drive', False)
                
                GoogleAuth.DEFAULT_SETTINGS['client_config_file'] = 'config/client_secrets.json'
                gauth = GoogleAuth()
                # Try to load saved client credentials
                gauth.LoadCredentialsFile("config/credentials.txt")
                if gauth.credentials is None:
                    gauth.LocalWebserverAuth() # Authenticate if they're not there
                elif gauth.access_token_expired:
                    gauth.Refresh() # Refresh them if expired
                else:
                    gauth.Authorize() # Initialize the saved creds
                # Save the current credentials to a file
                gauth.SaveCredentialsFile("config/credentials.txt")
                
                # -- WEBSCRAPE --
                drive = GoogleDrive(gauth)
                target_folder = drive.ListFile( # Target DOH folder ID
                    {'q': "'1_PhyL7788CLgZ717TklQ_iuMxvvnrrNn' in parents and trashed=false"}).GetList()
                webthreads = []
                progress = 0
                for idx, dataset in enumerate(target_folder):
                    # find datasets and put list its information
                    id = dataset['id'] # file ID 
                    query = 'case information' # look for this
                    title = dataset['title'].lower() # rename
                    if (query in title):
                        print(f'FOUND.. {title} : {idx}')
                        webthreads.append([title[52:], id, drive])
                    # update progress bar
                    progress = progress + int((1/len(target_folder))*100)
                    self.progress_signal.emit(progress, f'Retrieving Information... {title}', False)     
                # -- RETRIEVE --
                progress = 0
                for thread in webthreads:
                    title, id, drive = thread
                    print(f'dumping {thread} for {title}')
                    # dump file in the console
                    downloaded = drive.CreateFile({'id': id}) 
                    # splice filename then download file
                    downloaded.GetContentFile(f'{dir_dataset+title}') 
                    print(f'downloaded {title}')
                    # update progress bar
                    progress = progress + int((1/len(webthreads))*100)
                    self.progress_signal.emit(progress, f'Downloaded... {title}', False) 
                self.progress_signal.emit(0, 'Download Compelete', False) 
            # -- CONCATENATE --
            # emit a worker signal to the mainthread to reset progressbar
            self.progress_signal.emit(0,'Compiling...',False)
            # concatenate every scraped dataset to df
            folder = os.listdir(dir_dataset) # create folder object 
            # add store and read dataset information to list
            batch = []
            progress = 0
            for csv in folder:
                if '.csv' in csv:
                    # update progress bar
                    progress = progress + int((1/len(folder))*100)
                    self.progress_signal.emit(progress, f'Reading {csv}', False)
                    # append csv dump to batch list to concatenate                
                    batch.append(pd.read_csv(dir_dataset+csv, usecols=['DateRepConf'], parse_dates=['DateRepConf']))
            # concatenate csv
            df = pd.concat(batch, ignore_index=True)
            # check null values
            df.isnull().values.any()
            print(df,'\n')
            # rename date-confirmed column
            df.rename(columns={'DateRepConf':'Dates'}, inplace=True) 
            # count distinct-repeating dates
            df = df.groupby(df.Dates.dt.date).agg('count').rename(columns={'Dates':'Cases'})
            print(df,'\n')
            self.progress_signal.emit(100, f'Compression Successful  [{df.index[0]} — {df.index[-1]}]', True)
            self.output_signal.emit([df])
            
        elif self.index == 2: # forecast
            self.progress_signal.emit(0,'Initializing Forecast', False)
            # Statistic Library: SARIMAX Model
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            
            print(self.dataframe)
            in_train = self.dataframe[0].reset_index()
            in_test = self.dataframe[1].reset_index()
            integration = self.dataframe[2]
            predict = len(in_test)
            
            # check for stationarity (adf and pvalue)
            print('-- EVALUATING STATIONARITY --')
            progress = 0
            check_stationary = 0
            for i in range(integration):
                # update progress bar
                progress = progress + int((1/integration)*100) 
                check_stationary = in_train['Train'].diff().dropna()
                self.progress_signal.emit(progress, 'Evaluating Stationarity...', False)  
            # Statistic Library: Augmented Dickey Fuller Function
            from statsmodels.tsa.stattools import adfuller
            result_stationary = adfuller(check_stationary)
            adf = result_stationary[0]
            pvalue = result_stationary[1]
            print(f'ADF Statistic: {adf}')
            print(f'p-value: {pvalue}')

            # get aic scores
            min_params = (7,integration,8)
            # generate the sarimax instruction
            print('-- BUILDING ARIMA OBJECT --')
            self.progress_signal.emit(25, f'Building Object ARIMA{min_params}...', False)
            arima = SARIMAX(in_train['Train'], order=min_params, simple_differencing=False)
            # fitting the model
            print('-- FITTING THE MODEL --')
            self.progress_signal.emit(50, f'Fitting The Model... {arima}', False)
            model = arima.fit(disp=False)
            aic = model.aic
            print(model.summary())
            # forecast the model
            print('-- GENERATING MODEL FORECAST --')
            self.progress_signal.emit(75, f'Forecasting the Model... {model}', False)
            # retrieve forecast values
            predicted_values = model.get_prediction(end=model.nobs + predict).predicted_mean.round()
            # create new model column to existing dataframes
            out_train = in_train.assign(Model=predicted_values[:-predict-1])
            out_test = in_test.assign(Model=predicted_values[-predict-1:].reset_index(drop=True))
            # prints the error and accuracy in percent            
            print('-- SUMMARY --')
            import numpy as np
            err = np.subtract(out_test.Test, out_test.Model)
            abs_err = np.abs(err)
            total_cases = np.sum(out_test.Test)
            total_abs_err = np.sum(abs_err)
            
            mae = (total_abs_err/total_cases)*100
            accuracy = 100-(total_abs_err/total_cases)*100
            print('MAE: {:0.2f}%'.format(mae))
            print('Forecast Accuracy: {:0.2f}%'.format(accuracy))
            self.progress_signal.emit(100, f'Forecast Successful...', True)
            self.output_signal.emit([adf, pvalue, aic, min_params, mae, accuracy, out_test, model])
    
    def stop(self):
        self.is_running = False
        print('Stopping thread...', self.index)
        self.terminate()
class Plotter(FigureCanvas):
    # # -- HOW THIS CLASS WORKS --
    # # a figure instance to plot on
    # self.figure = plt.figure()
    # # this is the Canvas Widget that
    # # displays the 'figure'it takes the
    # # 'figure' instance as a parameter to __init__
    # self.canvas = FigureCanvas(self.figure)
    # # adding canvas to the layout
    # self.matplot_container.addWidget(self.canvas)
    
    def __init__(self, title, *dataframe, parent=None):
        super(Plotter, self).__init__(parent) 
        self.figure = None
        plt.close(self.figure)
        self.title = title
        
        # Creating your plot
        fig, ax = plt.subplots(figsize=(30, 10))
        
        # Plot the train and test sets on the same axis ax
        for obj in dataframe:
            obj.plot(ax=ax)
            
        fmt_month = mdates.MonthLocator() # Minor ticks every month.
        fmt_year = mdates.YearLocator() # Minor ticks every year.
        ax.xaxis.set_minor_locator(fmt_month)
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b')) # '%b' to get the names of the month
        ax.xaxis.set_major_locator(fmt_year)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        # fontsize for month labels
        ax.tick_params(labelsize=6, which='both')
        # create a second x-axis beneath the first x-axis to show the year in YYYY format
        sec_xaxis = ax.secondary_xaxis(-0.1)
        sec_xaxis.xaxis.set_major_locator(fmt_year)
        sec_xaxis.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        # Hide the second x-axis spines and ticks
        sec_xaxis.spines['bottom'].set_visible(False)
        sec_xaxis.tick_params(length=0, labelsize=6)

        plt.title(title, fontsize=7)
        plt.grid(which = 'both', linewidth=0.3)
        plt.xlabel('')
        plt.legend(loc='upper left') 
        
        self.figure = fig
        
    # def mouseDoubleClickEvent(self, event):
    #     print('matplot detached')
    #     plt.show()
        
        
try:
    # fixes the windows 11 tasbar icon
    import ctypes
    appid = 'mycompany.myproduct.subproduct.version' # arbitrary string
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(appid)
except:
    print("you're probably not on windows")
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = DYNARIMA()
    win.setWindowIcon(QtGui.QIcon(resource_path('DYNARIMA.ico')))
    win.show()
    sys.exit(app.exec_())

    