#include <esp_now.h>
#include <WiFi.h>

// Replace with ESP32_A's MAC address
uint8_t peer_mac[] = {0x8c, 0xbf, 0xea, 0x03, 0xc4, 0x28};

// Motor control pins (adjust as needed)
const int dirPin = 15;    // DIR pin
const int stepPin = 16;   // STEP pin
const int enablePin = 9; // ENABLE pin
const int ms1Pin = 46;    // MS1 pin
const int ms2Pin = 3;     // MS2 pin
const int ms3Pin = 8;     // MS3 pin

// Microstepping mode (change as needed)
const bool ms1State = HIGH; 
const bool ms2State = HIGH;
const bool ms3State = LOW; // Example: 1/8 microstepping

volatile bool start_operation = false;

// Callback to receive data from ESP32_A
void onDataRecv(const esp_now_recv_info *recv_info, const uint8_t *data, int len) {
  if (len == 1 && data[0] == 1) {
    start_operation = true;
  }
}

void setup() {
  Serial.begin(115200);
  
  // Initialize Wi-Fi in station mode
  // Set pin modes
  pinMode(dirPin, OUTPUT);
  pinMode(stepPin, OUTPUT);
  pinMode(enablePin, OUTPUT);
  pinMode(ms1Pin, OUTPUT);
  pinMode(ms2Pin, OUTPUT);
  pinMode(ms3Pin, OUTPUT);
  WiFi.mode(WIFI_STA);
  
  // Initialize ESP-NOW
  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init failed");
    return;
  }
  
  // Register ESP32_A as peer
  esp_now_peer_info_t peerInfo;
  memcpy(peerInfo.peer_addr, peer_mac, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;
  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add peer");
    return;
  }
  
  // Register receive callback
  esp_now_register_recv_cb(onDataRecv);

  // Set microstepping mode
  digitalWrite(ms1Pin, ms1State);
  digitalWrite(ms2Pin, ms2State);
  digitalWrite(ms3Pin, ms3State);

  // Enable the driver
  digitalWrite(enablePin, LOW); // LOW to enable, HIGH to disable
}

void loop() {
  if (start_operation) {
    Serial.println("Motor B starting open/close operation");
    
    // Open (clockwise)
    digitalWrite(dirPin, LOW);
    for (int i = 0; i < 700 * 8; i++) {
      digitalWrite(stepPin, HIGH);
      delayMicroseconds(200);
      digitalWrite(stepPin, LOW);
      delayMicroseconds(500);
    }
    delay(500);  // Pause for 1 second
    
    // Close (counterclockwise)
    digitalWrite(dirPin, HIGH);
    for (int i = 0; i < 700 * 8; i++) {
      digitalWrite(stepPin, HIGH);
      delayMicroseconds(200);
      digitalWrite(stepPin, LOW);
      delayMicroseconds(500);
    }
    
    // Send "done" signal to ESP32_A
    uint8_t data = 2;
    esp_err_t result = esp_now_send(peer_mac, &data, 1);
    if (result == ESP_OK) {
      Serial.println("Sent done signal to ESP32_A");
    } else {
      Serial.println("Failed to send done signal");
    }
    
    // Reset flag
    start_operation = false;
  }
}
