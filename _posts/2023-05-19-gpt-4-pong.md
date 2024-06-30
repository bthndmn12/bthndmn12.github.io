
---
layout: post
title:  "GPT-4 Pong"
date:   2023-05-19 12:45:00 +0100
categories: jekyll update

---

> Yazıya başlamadan önce Pong’un ne olduğundan bahsetmek istiyorum. Pong, 1972 yılında Atari firması tarafından piyasaya sürülen bir oyundur. Oyun tıpkı iki boyutlu bir tenis oyunu gibidir. Sağda ve solda iki raket ve ortada rastgele sağa ve sola zıplayan bir top. Sizi Pong’un sıkıcı tarihine sokmadan önce, bu basit ama bir dil modeli için harika kodlara hızlıca bir göz atalım.
> 
> Öncelikle yaklaşık 3 iterasyonda başarılı kod üretebildiğini belirtmeliyim. Her üç tekrarda da sağ raketi sol raketin üzerinde üretti ve aynı hatayı 2 kez verdi. Son, yani üçünçü iterasyonda başarılı oldu. Pong kodunu yazmamın sebebi basit olması ve iki yıl önce Arduino ile Processing kütüphanesini kullanarak küçük bir Pong oyunu yazmış olmamdır. Ancak burada sadece klavye fonsiyonlarını kullanacağız.
> 



İki raket, bir top ve skor tabelası için toplam 9 değişken oluşturuyorum.
```java
//Topun pozisyonu  
int xPos;  
int yPos;  
//Topun hızı  
int speed = 10;  
int vx = speed;  
int vy = speed;  
//Skor tabelası  
int countP1 = 0;  
int countP2 = 0;  
//Raketin konumu  
float xpos, ypos;         

GPT-4 ise 11 adet değişken oluşturuyor.

int paddleWidth = 20;  
int paddleHeight = 100;  
int paddleSpeed = 4;  
  
int leftPaddleX;  
int leftPaddleY;  
int rightPaddleX;  
int rightPaddleY;  
  
int ballSize = 20;  
int ballX;  
int ballY;  
int ballSpeedX = 3;  
int ballSpeedY = 2;
```

---


Benim setup() metodum

```java
void setup() {  
  
  fullScreen();  
  frameRate(60);  
  noStroke();        
  noCursor();  
  xPos = width/2;  
  yPos = height/2;  
}

GPT-4 setup() metodu

void setup() {  
  size(800, 600);  
  noStroke();  
  
  leftPaddleX = 30;  
  leftPaddleY = height / 2 - paddleHeight / 2;  
  rightPaddleX = width - 30 - paddleWidth;  
  rightPaddleY = height / 2 - paddleHeight / 2;  
    
  ballX = width / 2;  
  ballY = height / 2;  
}
```


_Yukarıdaki kodlardaki benzerliklere bakacak olursak, ikisinde de noStroke() metodu kullanılmış ve topun ilk başlama noktasını ekranın ortası yapılmış._

---

Benim draw() metodum
```java
void draw() {  
  background(0);  
  
// Raketlerin, topun ve skor tabelasını oluşturma  
 if (start == true) {  
    textSize(100);  
    text(countP1/12, 100, 100);  
    text(countP2/12, 1700, 100);  
    rect(1820, xpos, 30, 200, 10);  
    rect(100, ypos, 30, 200, 10);  
    circle(xPos, yPos, 20);  
    stroke(255);  
    line(width/2, 0, width/2, height);  
  }  
  
//Topun yönünü tayin etme  
  xPos+=vx;  
  yPos+=vy;  
  
  /*Aşağıdaki karar yapılarında kısaca eğer top belirtilen   
yükseklik ve genişlik değerlerinin arasında sahip olduğu yönün tam tersine  
gitmesin sağlıyor. Skorun arttığı yerlerde ise o belirtilen değer sahip köşe-  
lerden geçince puanın artmasını sağlıyor.  
*/  
  if ((yPos <= height+speed/2 && yPos >= height) || (yPos <=speed && yPos >= 0)) {  
    vy= -vy;  
    pulse.stop();  
  }  
  if (((xPos <= 130) && (xPos >=99)) && ((yPos <= ypos+200) && ((yPos >=ypos)))) {  
    vx= -vx;  
    pulse.play();  
  }  
  if (((xPos <= 1850) && (xPos >=1820)) && ((yPos <= xpos+200) && ((yPos >=xpos)))) {  
    vx= -vx;  
    pulse.play();  
  }  
  if ((xPos >= 1830)) {  
    countP1++;  
  }  
  if ((xPos >= 0)&&(xPos<=130)) {  
    countP2++;  
  }  
  if ((xPos > 150)&&(xPos<1800)) {  
    pulse.stop();  
  }  
  if ((xPos <= 20)&&(xPos>=0)||(xPos >= 1900)&&(xPos<=1920)) {  
    xPos = width/2;  
    yPos = height/2;  
  }  
  if (countP1/12 == 10 && countP2/12<10) {  
    start = false;  
    text("P1 WINS", 100, 100);  
  }  
  if (countP2/12 == 10 && countP1/12<10) {  
    start = false;  
    text("P2 WINS", 100, 100);  
  }  
}
```

GPT-4 draw() metodu

```java
void draw() {  
  background(0);  
// Keep paddles inside the screen.  
  leftPaddleY = constrain(leftPaddleY, 0, height - paddleHeight);  
  rightPaddleY = constrain(rightPaddleY, 0, height - paddleHeight);  
  
  // Update ball position.  
  ballX += ballSpeedX;  
  ballY += ballSpeedY;  
  
  // Bounce ball off top and bottom of the screen.  
  if (ballY < 0 || ballY > height - ballSize) {  
    ballSpeedY = -ballSpeedY;  
  }  
  
  // Bounce ball off paddles.  
  if ((ballX <= leftPaddleX + paddleWidth && ballY + ballSize >= leftPaddleY && ballY <= leftPaddleY + paddleHeight) || (ballX + ballSize >= rightPaddleX && ballY + ballSize >= rightPaddleY && ballY <= rightPaddleY + paddleHeight)) {  
    ballSpeedX = -ballSpeedX;  
  }  
  
  // Reset ball if it goes off the screen.  
  if (ballX < 0 || ballX > width) {  
    ballX = width / 2;  
    ballY = height / 2;  
    ballSpeedX = -ballSpeedX;  
  }  
  
  // Draw paddles and ball.  
  fill(255);  
  rect(leftPaddleX, leftPaddleY, paddleWidth, paddleHeight);  
  rect(rightPaddleX, rightPaddleY, paddleWidth, paddleHeight);  
  ellipse(ballX, ballY, ballSize, ballSize);  
}
```

- _İki kod arasında dikkatimi ilk çeken şey background(0) oldu. Fakat haliyle klasikleşmiş bir oyun olduğu için Pong’un arka planı siyah olmalıydı._
- _Bir farklılık olarak benim raketleri içeride tutmaya yarayan bir kodum yok._
- _İkinci benzerlik topun hareketinin güncellenmesi fakat bunun içinde spesifik bir şey yazılamazdı kanaatimce._
- _Üçüncü benzerlik ise eğer top sağ ve sol köşelerden çıktığı takdirde ortadan tekrar başlaması. Bu da Pong’un bir diğer özelliği._

![](https://cdn-images-1.medium.com/max/800/1*x7rAc_YReVecjvO7mylZ8A.png)

Sohbete ait bir Ekran Görüntüsü

Github ve çoğu kod paylaşımı yapılan sitelerdeki Processing ile yazılmış Pong oyunları bu ve buna benzer kodlar. Hem benim hem de GPT-4’ün yapmış olduğu Pong’un Github linklerini ve videolarını aşağıya ekledim.

Benim yapmış olduğum Pong

GPT-4'ün yapmış olduğu Pong

Github Linkleri ve sohbetin linki:

[**dmn_ProcessingandArduinoGames/realPongProcessing.pde at main ·…**  
_Basic Atari Breakout game using Processing and Arduino - dmn_ProcessingandArduinoGames/realPongProcessing.pde at main ·…_github.com](https://github.com/bthndmn12/dmn_ProcessingandArduinoGames/blob/main/ArduinoPong/realPongProcessing/realPongProcessing.pde "https://github.com/bthndmn12/dmn_ProcessingandArduinoGames/blob/main/ArduinoPong/realPongProcessing/realPongProcessing.pde")[](https://github.com/bthndmn12/dmn_ProcessingandArduinoGames/blob/main/ArduinoPong/realPongProcessing/realPongProcessing.pde)

[**dmn_ProcessingandArduinoGames/GPT4Pong.pde at main · bthndmn12/dmn_ProcessingandArduinoGames**  
_Basic Atari Breakout game using Processing and Arduino - dmn_ProcessingandArduinoGames/GPT4Pong.pde at main ·…_github.com](https://github.com/bthndmn12/dmn_ProcessingandArduinoGames/blob/main/ArduinoPong/GPT4Pong.pde "https://github.com/bthndmn12/dmn_ProcessingandArduinoGames/blob/main/ArduinoPong/GPT4Pong.pde")[](https://github.com/bthndmn12/dmn_ProcessingandArduinoGames/blob/main/ArduinoPong/GPT4Pong.pde)

[**dmn_ProcessingandArduinoGames/GPT4Chat.txt at main · bthndmn12/dmn_ProcessingandArduinoGames**  
_Basic Atari Breakout game using Processing and Arduino - dmn_ProcessingandArduinoGames/GPT4Chat.txt at main ·…_github.com](https://github.com/bthndmn12/dmn_ProcessingandArduinoGames/blob/main/ArduinoPong/GPT4Chat.txt "https://github.com/bthndmn12/dmn_ProcessingandArduinoGames/blob/main/ArduinoPong/GPT4Chat.txt")[](https://github.com/bthndmn12/dmn_ProcessingandArduinoGames/blob/main/ArduinoPong/GPT4Chat.txt)