import sys
import subprocess as sp
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QFormLayout, QLabel,
    QLineEdit, QSpinBox, QCheckBox, QMessageBox, QDoubleSpinBox
)

class PlayWid(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Play')
        self.setGeometry(100, 100, 400, 300)

        lvl1 = QPushButton('Level 1', self)
        lvl2 = QPushButton('Level 2', self)
        lvl3 = QPushButton('Level 3', self)
        lvl4 = QPushButton('Level 4', self)
        lvl5 = QPushButton('Level 5', self)

        typ = [lvl1,lvl2,lvl3,lvl4,lvl5]
        layout = QVBoxLayout()

        for button in typ:
            layout.addWidget(button)

        self.setLayout(layout)

        lvl1.clicked.connect(self.ch_lvl1)
        lvl2.clicked.connect(self.ch_lvl2)
        lvl3.clicked.connect(self.ch_lvl3)
        lvl4.clicked.connect(self.ch_lvl4)
        lvl5.clicked.connect(self.ch_lvl5)

    def ch_lvl1(self):
        command = ['py', '-3.9', '.\\agent.py', '--player', '1', '--cordinates', '60', '60', 
                   '--optimization', '1', '--famine', '0', '--load_model', 'snakes_base.pth', 
                   '--save_name', 'model.pth', '--h', '980', '--w', '820', '--snake_count', '1', '--food_count', '5', '--win', '1']
        res = sp.run(command, shell=True, capture_output=True, text=True)
        if res.returncode == 0:
            QMessageBox.information(self, 'Congratulations', 'You have won Level 1!')
    
    def ch_lvl2(self):
        command = ['py', '-3.9', '.\\agent.py', '--player', '1', '--cordinates', '60', '60', 
                   '--optimization', '1', '--famine', '1', '--load_model', 'snakes_base.pth', 
                   '--save_name', 'model.pth', '--h', '1200', '--w', '920', '--snake_count', '2', '--food_count', '4', '--win', '1']
        res = sp.run(command, shell=True, capture_output=True, text=True)
        if res.returncode == 0:
            QMessageBox.information(self, 'Congratulations', 'You have won Level 2!')
    
    def ch_lvl3(self):
        command = ['py', '-3.9', '.\\agent.py', '--player', '1', '--cordinates', '60', '60', 
                   '--optimization', '1', '--famine', '1', '--load_model', 'snake_medium.pth', 
                   '--save_name', 'model.pth', '--h', '1920', '--w', '1080', '--snake_count', '10', '--food_count', '15', '--win', '1']
        res = sp.run(command, shell=True, capture_output=True, text=True)
        if res.returncode == 0:
            QMessageBox.information(self, 'Congratulations', 'You have won Level 3!')
    
    def ch_lvl4(self):
        command = ['py', '-3.9', '.\\agent.py', '--player', '1', '--cordinates', '60', '60', 
                   '--optimization', '1', '--famine', '1', '--load_model', 'snake_medium.pth', 
                   '--save_name', 'model.pth', '--h', '1920', '--w', '1080', '--snake_count', '15', '--food_count', '10', '--win', '1']
        res = sp.run(command, shell=True, capture_output=True, text=True)
        if res.returncode == 0:
            QMessageBox.information(self, 'Congratulations', 'You have won Level 4!')
    
    def ch_lvl5(self):
        command = ['py', '-3.9', '.\\agent.py', '--player', '1', '--cordinates', '60', '60', 
                   '--optimization', '1', '--famine', '1', '--load_model', 'snake_medium.pth', 
                   '--save_name', 'model.pth', '--h', '1920', '--w', '1080', '--snake_count', '15', '--food_count', '10', '--win', '1']
        res = sp.run(command, shell=True, capture_output=True, text=True)
        if res.returncode == 0:
            QMessageBox.information(self, 'Congratulations', 'You have won Level 5!')
    
    def exitClicked(self):
        sys.exit()


class TrainWid(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Train')
        self.setGeometry(100, 100, 600, 400)

        main_layout = QVBoxLayout()
        form_layout = QFormLayout()

        self.player_edit = QCheckBox('Player Mode')
        self.player_edit.setChecked(True)  
        self.coordinates_edit = QLineEdit('60,60')
        self.optimization_edit = QCheckBox('Optimization')
        self.optimization_edit.setChecked(True)  
        self.famine_edit = QCheckBox('Famine')
        self.load_model_edit = QLineEdit('snake_medium.pth')
        self.save_name_edit = QLineEdit('model.pth')
        self.height_edit = QSpinBox()
        self.height_edit.setRange(100, 3000)
        self.height_edit.setValue(1920)
        self.width_edit = QSpinBox()
        self.width_edit.setRange(100, 3000)
        self.width_edit.setValue(1080)
        self.snake_count_edit = QSpinBox()
        self.snake_count_edit.setRange(1, 10000)
        self.snake_count_edit.setValue(10)
        self.food_count_edit = QSpinBox()
        self.food_count_edit.setRange(1, 10000)
        self.food_count_edit.setValue(15)
        self.win_edit = QCheckBox('Winnable')
        self.win_edit.setChecked(True) 
        #conf
        self.speed_edit = QSpinBox()
        self.speed_edit.setRange(1, 2147483647)
        self.speed_edit.setValue(30)
        self.lr_edit = QDoubleSpinBox()
        self.lr_edit.setRange(0.001, 1.0)
        self.lr_edit.setValue(0.001)
        self.lr_edit.setDecimals(4)
        self.max_mem_edit = QSpinBox()
        self.max_mem_edit.setRange(1, 2147483647)
        self.max_mem_edit.setValue(10000)
        self.batch_size_edit = QSpinBox()
        self.batch_size_edit.setRange(1, 2147483647)
        self.batch_size_edit.setValue(5000)
        self.epsilon_edit = QSpinBox()
        self.epsilon_edit.setRange(1, 100)
        self.epsilon_edit.setValue(50)
        self.gamma_edit = QDoubleSpinBox()
        self.gamma_edit.setRange(0.0, 1.0)
        self.gamma_edit.setValue(0.978)
        self.gamma_edit.setDecimals(4)
        #---end conf

        form_layout.addRow('Player Mode:', self.player_edit)
        form_layout.addRow('Coordinates (x,y):', self.coordinates_edit)
        form_layout.addRow('Optimization:', self.optimization_edit)
        form_layout.addRow('Famine:', self.famine_edit)
        form_layout.addRow('Load Model:', self.load_model_edit)
        form_layout.addRow('Save Name:', self.save_name_edit)
        form_layout.addRow('Height:', self.height_edit)
        form_layout.addRow('Width:', self.width_edit)
        form_layout.addRow('Snake Count:', self.snake_count_edit)
        form_layout.addRow('Food Count:', self.food_count_edit)
        form_layout.addRow('Win Condition:', self.win_edit) 

        #conf
        form_layout.addRow('Speed:', self.speed_edit)
        form_layout.addRow('LR:', self.lr_edit)
        form_layout.addRow('Max memory:', self.max_mem_edit)
        form_layout.addRow('Batch size:', self.batch_size_edit)
        form_layout.addRow('Epsilon:', self.epsilon_edit)
        form_layout.addRow('Gamma:', self.gamma_edit)
        #---end conf
        main_layout.addLayout(form_layout)

        play_button = QPushButton('Play', self)
        play_button.clicked.connect(self.playClicked)
        main_layout.addWidget(play_button)

        self.coordinates = self.coordinates_edit.text().split(',')
        self.setLayout(main_layout)
    
    def formatConf(self):
        conf_str = f"""C_SPEED = {self.speed_edit.value()}
C_MAX_MEMORY = {self.max_mem_edit.value()}
C_BATCH_SIZE = {self.batch_size_edit.value()}
C_LR = {self.lr_edit.value()}
C_EPSILON = {self.epsilon_edit.value()}
C_GAMMA = {self.gamma_edit.value()}
"""
        with open("conf.py", 'w') as f:
            f.write(conf_str)

    def playClicked(self):
        self.formatConf()
        if len(self.coordinates) != 2:
            QMessageBox.critical(self, 'Error', 'Please enter coordinates in the format "x,y"')
            return
        command = [
            'py', '-3.9', './agent.py',
            '--player', '1' if self.player_edit.isChecked() else '0',
            '--cordinates', self.coordinates[0], self.coordinates[1],
            '--optimization', '1' if self.optimization_edit.isChecked() else '0',
            '--famine', '1' if self.famine_edit.isChecked() else '0',
            '--load_model', self.load_model_edit.text(),
            '--save_name', self.save_name_edit.text(),
            '--h', str(self.height_edit.value()),
            '--w', str(self.width_edit.value()),
            '--snake_count', str(self.snake_count_edit.value()),
            '--food_count', str(self.food_count_edit.value()),
            '--win', '1' if self.win_edit.isChecked() else '0'
        ]

        res = sp.run(command, shell=True, capture_output=True, text=True)
        if res.returncode == 0:
            QMessageBox.information(self, 'Success', 'You won!')
        else:
            QMessageBox.critical(self, 'Error', f'An error occurred:\n{res.stderr}')
   


class SimpleApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.play_window = None
        self.train_window = None

    def initUI(self):
        self.setWindowTitle('SnakesAI')
        self.setGeometry(100, 100, 400, 300)

        play_button = QPushButton('Play', self)
        train_button = QPushButton('Train', self)
        exit_button = QPushButton('Exit', self)

        layout = QVBoxLayout()
        layout.addWidget(play_button)
        layout.addWidget(train_button)
        layout.addWidget(exit_button)

        self.setLayout(layout)

        play_button.clicked.connect(self.playClicked)
        train_button.clicked.connect(self.trainClicked)
        exit_button.clicked.connect(self.exitClicked)

    def playClicked(self):
        if not self.play_window:
            self.play_window = PlayWid()
        self.play_window.show()

    def trainClicked(self):
        if not self.train_window:
           self.train_window = TrainWid()
        self.train_window.show()

    def exitClicked(self):
        sys.exit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SimpleApp()
    window.show()
    sys.exit(app.exec())