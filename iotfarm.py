import sys
sys.path.append('/home/pi/pySX127x')

from SX127x.LoRa import LoRa
from SX127x.board_config import BOARD
from SX127x.constants import MODE
from time import sleep
import sys
import datetime

BOARD.setup()

class LoRaRcvCont(LoRa):
    def __init__(self, verbose=False):
        super(LoRaRcvCont, self).__init__(verbose)
        self.set_mode(MODE.SLEEP)
        self.set_dio_mapping([0] * 6)

    def start(self):
        self.reset_ptr_rx()
        self.set_mode(MODE.RXCONT)
        print("LoRa receiver started...")

        try:
            while True:
                sleep(0.3)
                sys.stdout.flush()
        except KeyboardInterrupt:
            print("\nStopping LoRa receiver...")
            self.set_mode(MODE.SLEEP)
            BOARD.teardown()

    def on_rx_done(self):
        print("\n====================================")
        print("Packet received at", datetime.datetime.now())
        print("====================================")
        self.clear_irq_flags(RxDone=1)
        payload = self.read_payload(nocheck=False)
        try:
            data = bytes(payload).decode("utf-8")
        except:
            data = bytes(payload).decode("latin-1", "ignore")
        
        print("Raw data:", data)
        print("Payload bytes:", payload)
        
   
        if "Temp:" in data and "Humidity:" in data:
            try:
                temp_str = data.split("Temp:")[1].split(";")[0]
                humidity_str = data.split("Humidity:")[1].split(";")[0]
                temp = float(temp_str)
                humidity = float(humidity_str)
                
                print(f"Nhiệt độ: {temp} °C")
                print(f"Độ ẩm: {humidity} %")
            except Exception as e:
                print("Không thể phân tích dữ liệu:", e)
        else:
            print("Dữ liệu không chứa thông tin nhiệt độ và độ ẩm.")
        
        print("------------------------------------")
        print("Packet RSSI:", self.get_pkt_rssi_value(), "dBm")
        print("Packet SNR :", self.get_pkt_snr_value(), "dB")
        print("Frequency  :", self.get_freq() / 1e6, "MHz")
        print("Length     :", len(payload), "bytes")
        print("====================================\n")

if __name__ == "__main__":
    lora = LoRaRcvCont(verbose=False)
    lora.start()
