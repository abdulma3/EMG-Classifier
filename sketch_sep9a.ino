void setup() {

  Serial.begin(9600);  // start serial communication at 9600 baud (bits per second)

}

void loop() {

  int emgValue = analogRead(A0);  // read the voltage on pin A0 as a number 0-1023

  Serial.println(emgValue);       // print it, one value per line

  delay(5);                        // small pause so we don't flood the serial port

}
