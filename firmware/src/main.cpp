// NoManual rig firmware entry point (ESP32, Arduino framework).
//
// All machine and protocol logic lives in the Arduino-free controller (controller.cpp), which
// the host tests exercise. This file only wires it to the hardware: DIP switches, UART, servo
// PWM, the panel LED and the last servo position kept in NVS.
//
// NOTHING HERE HAS RUN ON A REAL BOARD YET (cross-compiled only, 2026-10-27).
#include <Arduino.h>
#include <ESP32Servo.h>
#include <Preferences.h>

#include "configs.h"
#include "controller.h"
#include "rig_pins.h"

namespace {

nm::Controller g_ctl;
Servo g_servo_u;
Servo g_servo_v;
Preferences g_prefs;
bool g_servos_attached = false;
int g_last_us_u = -1;
int g_last_us_v = -1;
double g_saved_u = 0.5;
double g_saved_v = 0.5;

// Reads the DIP switches three times 5 ms apart; returns -1 if the reading is not stable.
int read_draw() {
  for (uint8_t pin : rig::DIP_PINS) pinMode(pin, INPUT_PULLUP);
  delay(20);
  int first = -1;
  for (int attempt = 0; attempt < 3; ++attempt) {
    int value = 0;
    for (int bit = 0; bit < 4; ++bit) {
      if (digitalRead(rig::DIP_PINS[bit]) == LOW) value |= 1 << bit;
    }
    if (first >= 0 && value != first) return -1;
    first = value;
    delay(5);
  }
  return first;
}

void load_saved_position() {
  g_prefs.begin("nm", true);
  g_saved_u = g_prefs.getDouble("u", 0.5);
  g_saved_v = g_prefs.getDouble("v", 0.5);
  g_prefs.end();
  if (!(g_saved_u >= 0.0 && g_saved_u <= 1.0)) g_saved_u = 0.5;
  if (!(g_saved_v >= 0.0 && g_saved_v <= 1.0)) g_saved_v = 0.5;
}

// Keeps the last settled servo position, so the next power-up glides from (about) where the
// servos really are instead of jumping. Written only when it moved, at most once per action.
void save_position(double u, double v) {
  if (fabs(u - g_saved_u) < 0.002 && fabs(v - g_saved_v) < 0.002) return;
  g_prefs.begin("nm", false);
  g_prefs.putDouble("u", u);
  g_prefs.putDouble("v", v);
  g_prefs.end();
  g_saved_u = u;
  g_saved_v = v;
}

void write_servos() {
  if (!g_servos_attached) return;
  const int us_u = rig::servo_us(g_ctl.out_u());
  const int us_v = rig::servo_us(g_ctl.out_v());
  if (us_u != g_last_us_u) {
    g_servo_u.writeMicroseconds(us_u);
    g_last_us_u = us_u;
  }
  if (us_v != g_last_us_v) {
    g_servo_v.writeMicroseconds(us_v);
    g_last_us_v = us_v;
  }
}

void flush_replies() {
  char line[nm::MAX_LINE + 2];
  while (g_ctl.pop_line(line, sizeof line)) {
    Serial.write(reinterpret_cast<const uint8_t*>(line), strlen(line));
    Serial.write('\n');
  }
}

}  // namespace

void setup() {
  pinMode(rig::LED_PIN, OUTPUT);
  digitalWrite(rig::LED_PIN, LOW);
  Serial.begin(115200);

  const int draw = read_draw();
  const nm_config_t* cfg = (draw >= 1 && draw <= NM_N_CONFIGS) ? &NM_CONFIGS[draw - 1] : nullptr;
  load_saved_position();
  g_ctl.begin(draw, cfg, g_saved_u, g_saved_v, millis());

  if (g_ctl.state() != nm::State::Halted) {
    // First pulse = the remembered position; the controller then glides to the start.
    g_servo_u.setPeriodHertz(rig::SERVO_PERIOD_HZ);
    g_servo_v.setPeriodHertz(rig::SERVO_PERIOD_HZ);
    g_servo_u.attach(rig::SERVO_U_PIN, rig::SERVO_US_AT_0, rig::SERVO_US_AT_1);
    g_servo_v.attach(rig::SERVO_V_PIN, rig::SERVO_US_AT_0, rig::SERVO_US_AT_1);
    g_servos_attached = true;
    write_servos();
  }
  flush_replies();
}

void loop() {
  const uint32_t now = millis();
  while (Serial.available() > 0) {
    const int b = Serial.read();
    if (b < 0) break;
    g_ctl.feed(static_cast<uint8_t>(b), now);
  }
  g_ctl.update(millis());
  flush_replies();
  write_servos();
  digitalWrite(rig::LED_PIN, g_ctl.led(millis()) ? HIGH : LOW);
  if (g_ctl.take_settled()) save_position(g_ctl.out_u(), g_ctl.out_v());
}
