#include <Encoder.h>

Encoder knob(2, 3);

int ledPin = 11; 
long pos = 0;
const long minPos = 0;
const long maxPos = 120; 

void setup() {
  Serial.begin(9600);
  pinMode(ledPin, OUTPUT);
}

void loop() {
  long newPos = -knob.read();

  if (newPos < minPos) {
    newPos = minPos;
    knob.write(-minPos);
  } else if (newPos > maxPos) {
    newPos = maxPos;
    knob.write(-maxPos);
  }

  if (newPos != pos) {
    pos = newPos;
    Serial.print("Encoder Position: ");
    Serial.println(pos);

    int brightness = map(pos, minPos, maxPos, 0, 255);

    analogWrite(ledPin, brightness);
  }

  delay(10); 
}