
#include <esp_now.h>
#include <Arduino.h>
#include <WiFi.h>

// Define LED pins (assuming these are defined based on your setup)
const int greenLED = 41;  // Adjust pin numbers as per your hardware
const int yellowLED = 38;
//const int redLED = 40;

// Define stepper motor pins
const int dirPin = 6;    // DIR pin
const int stepPin = 7;   // STEP pin
const int enablePin = 3; // ENABLE pin
const int ms1Pin = 17;   // MS1 pin
const int ms2Pin = 16;   // MS2 pin
const int ms3Pin = 15;   // MS3 pin
const int msvdd = 42;    // Power pin (if needed)

// Microstepping mode (1/16 microstepping)
const bool ms1State = HIGH;
const bool ms2State = HIGH;
const bool ms3State = HIGH;

// Steps per cm (based on 8000 steps = 10 cm)
const int stepsPerCm = 800;

// Track position (in steps relative to initial position)
long currentPosition = 0; // Use long to handle large step counts

volatile bool b_finished = false;


// Replace with ESP32_B's MAC address
uint8_t peer_mac[] = {0x8c, 0xbf, 0xea, 0x03, 0xbb, 0x08};

// Define TX and RX pins for UART (change if needed)
#define TXD1 43
#define RXD1 44

// Use Serial1 for UART communication with Raspberry Pi
HardwareSerial mySerial(2);


// Callback to receive "done" signal from ESP32_B
void onDataRecv(const esp_now_recv_info *recv_info, const uint8_t *data, int len) {
  if (len == 1 && data[0] == 2) {
    b_finished = true;
  }
}

void setup() {
  // Initialize serial communication for debugging
  Serial.begin(115200);

  // Set LED pin modes
  pinMode(greenLED, OUTPUT);
  pinMode(yellowLED, OUTPUT);
  digitalWrite(greenLED, HIGH);
  digitalWrite(yellowLED, HIGH);

  // Set stepper motor pin modes
  pinMode(dirPin, OUTPUT);
  pinMode(stepPin, OUTPUT);
  pinMode(enablePin, OUTPUT);
  pinMode(ms1Pin, OUTPUT);
  pinMode(ms2Pin, OUTPUT);
  pinMode(ms3Pin, OUTPUT);
  pinMode(msvdd, OUTPUT);

  // Set microstepping mode
  digitalWrite(ms1Pin, ms1State);
  digitalWrite(ms2Pin, ms2State);
  digitalWrite(ms3Pin, ms3State);

  // Enable the stepper driver
  digitalWrite(enablePin, LOW); // LOW to enable
  digitalWrite(msvdd, HIGH);    // Power to driver (if needed)

  // Initialize Wi-Fi in station mode for ESP-NOW
  WiFi.mode(WIFI_STA);

  // Initialize ESP-NOW
  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init failed");
    return;
  }

  // Register ESP32_B as peer
  esp_now_peer_info_t peerInfo;
  memcpy(peerInfo.peer_addr, peer_mac, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;
  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add peer");
    return;
  }

  // Register receive callback for ESP-NOW
  esp_now_register_recv_cb(onDataRecv);
}

void moveMotor(int distanceCm, bool direction) {
  // direction: LOW = forward (away from start), HIGH = backward (toward start)
  digitalWrite(dirPin, direction);
  int steps = distanceCm * stepsPerCm; // Convert cm to steps

  for (int i = 0; i < steps; i++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(20);  // Pulse width
    digitalWrite(stepPin, LOW);
    delayMicroseconds(500); // Step interval
  }

  // Update position
  if (direction == LOW) {
    currentPosition += steps; // Moving away from start
  } else {
    currentPosition -= steps; // Moving back toward start
  }
}

void returnToStart() {
  if (currentPosition > 0) {
    digitalWrite(yellowLED, HIGH); // Indicate returning
    moveMotor(currentPosition / stepsPerCm, HIGH); // Reverse direction
    digitalWrite(yellowLED, LOW);
  }
  currentPosition = 0; // Reset position to 0
}

void performMission(int distanceCm) {
  digitalWrite(greenLED, HIGH);     // Indicate mission start
  moveMotor(distanceCm, LOW);       // Move forward
  Serial.println("Motor A moved forward " + String(distanceCm) + " cm");
  delay(400);

  uint8_t data = 1;                 // "start" signal
  esp_err_t result = esp_now_send(peer_mac, &data, 1);
  if (result == ESP_OK) {
    Serial.println("Sent start signal to ESP32_B");
  } else {
    Serial.println("Failed to send start signal");
  }

  while (!b_finished) {             // Wait for "done" signal from ESP32_B
    delay(10);
  }

  returnToStart();
  Serial.println("Motor A returned to start");
  b_finished = false;
  digitalWrite(greenLED, LOW);      // Mission complete
}






void loop() {
    if (Serial.available()) {
        String received = Serial.readStringUntil('\n');
        int value = received.toInt();
        // Turn off all LEDs first
        digitalWrite(greenLED, LOW);
        digitalWrite(yellowLED, LOW);

        // Control LEDs based on received string
        if (value == 0) {
            performMission(0);
            return;
        } else if (value == 1) {
            performMission(20);
            return;
        } else if (value == 2) {
            performMission(44);
            return;
        } else if (value == 3) {
            performMission(62);
            return;
        }
        delay(100); // Delay before next command
        Serial.println(received);  // For debugging
        Serial.println(value);  // For debugging
    }
}
