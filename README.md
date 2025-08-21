## Structure:

```
/chess-app
├── predictors/
│   ├── __init__.py
│   ├── base_predictor.py
│   ├── random_predictor.py
│   └── stockfish_predictor.py
├── static/
│   ├── css/
│   │   └── chessboard-1.0.0.min.css
│   ├── img/
│   │   └── chesspieces/
│   │       └── wikipedia/
│   │           ├── bB.png
│   │           ├── bK.png
│   │           ├── bN.png
│   │           ├── bP.png
│   │           ├── bQ.png
│   │           ├── bR.png
│   │           ├── wB.png
│   │           ├── wK.png
│   │           ├── wN.png
│   │           ├── wP.png
│   │           ├── wQ.png
│   │           └── wR.png
│   └── js/
│       ├── chessboard-1.0.0.min.js
│       └── jquery-3.7.1.min.js
├── templates/
│   └── index.html
├── app.py
└── requirements.txt
```

### Dependencies

- [jquery-3.7.1.min.js](https://jquery.com/download/)
- [chessboard.js](https://chessboardjs.com/index.html#download)