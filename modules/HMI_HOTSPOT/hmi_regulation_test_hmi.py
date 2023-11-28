#
#   Hello World client in Python
#   Connects REQ socket to tcp://localhost:5555
#   Sends "Hello" to server, expects "World" back
#

import zmq
import time
import threading
import json

import sys
# 1. Import `QApplication` and all the required widgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QLineEdit


class MessageClient:

    def __init__(self,port_hmi,wsf):
        self.wsf = wsf
        wsf.ms = self
        self.port_hmi = port_hmi

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:"+str(self.port_hmi))
        self.receive_message()

        
    def send_message(self,message):
        jmessage = json.dumps(message)
        self.socket.send_json(jmessage)
        msg = self.socket.recv_json()
        print('MESSAGE RECEIVED:', msg)
        self.receive_message()


    def receive_message(self):
        jmessage = json.dumps("Ready to receive request")
        self.socket.send_json(jmessage)
        #while True:
        #  Wait for next request from client
        jmsg = self.socket.recv_json()
        msg = json.loads(jmsg)
        self.message_received(msg)
        

    def message_received(self,message):
        wsf.receive_options(message)

    class Letter(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)




class WindowSelectFP:

    def __init__(self):#,layout,message_server):
        #self.layout = layout
        #self.message_server=message_server

        self.max_options = -1
        self.message_server = None

        self.window = QWidget()
        self.window.setWindowTitle('Info from Mercury: Select FP')
        self.layout = QVBoxLayout()

        self.lineEntry = QLineEdit()
        self.layout.addWidget(self.lineEntry)
        btn = QPushButton('Selection')
        btn.clicked.connect(self.send_reply)  # Connect clicked to greeting()
        self.layout.addWidget(btn)

        self.window.setLayout(self.layout)
        #window.show()


    def send_reply(self):
        option = int(self.lineEntry.text())
        if option>self.max_options:
            self.lineEntry.setText("")
        else:
            msg = ms.Letter()
            msg['from'] = self.fr
            msg['to'] = self.to
            msg['type'] = 'reply_option_selected'
            msg['body'] = {'option_selected_fp':option,'flight_uid':self.flight_uid}
            ms.send_message(msg)


    def receive_options(self,msg):
        print(msg)
        self.to = msg['from']
        self.fr = msg['to']
        self.flight_uid = int(msg['body']['flight_uid'])
        cost_options = msg['body']['cost_options']
        costs = msg['body']['cost_options']['total_cost']

        self.max_options = len(costs)

        for i in range(len(costs)):
            message = "Option "+str(i)+" -- Total cost:"+str(costs[i])+"-- Fuel cost:"+str(cost_options['fuel_cost'][i])+\
                "-- Delay cost:"+str(cost_options['delay_cost'][i])+"-- CRCO cost:"+str(cost_options['crco_cost'][i])

            self.layout.addWidget(QLabel(message))



if __name__=="__main__":
    port_hmi = int(input('Enter port HMI: '))
    
    app = QApplication(sys.argv)

    wsf = WindowSelectFP()

    wsf.window.show()

    ms = MessageClient(port_hmi,wsf)
    
    sys.exit(app.exec_())






    
    #for request in range(10)   
        #  Do some 'work'
    #    time.sleep(1)
