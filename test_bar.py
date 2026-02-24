import sys
from PyQt5.QtWidgets import QApplication, QProgressBar, QWidget, QVBoxLayout

app = QApplication(sys.argv)
app.setStyleSheet("""
QProgressBar { background: #333; border: 1px solid #444; border-radius: 6px; text-align: center; }
QProgressBar::chunk { border-radius: 4px; }
QProgressBar[class="bar-red"]::chunk { background-color: #f44336; }
QProgressBar[class="bar-green"]::chunk { background-color: #4CAF50; }
""")

w = QWidget()
l = QVBoxLayout(w)
b1 = QProgressBar()
b1.setRange(0, 100)
b1.setValue(50)
b1.setProperty("class", "bar-red")
l.addWidget(b1)
w.show()
sys.exit(0)
