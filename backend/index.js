const dotenv = require('dotenv');
dotenv.config();

const express = require('express');
const cors = require('cors');
const cookieParser = require('cookie-parser');
const passport = require('passport');
const compression = require('compression');
const connectDB = require('./lib/db');
const customGptRoutes = require('./routes/customGptRoutes');
const authRoutes = require('./routes/authRoutes');
const invitationRoutes = require('./routes/invitationRoutes');
const chatHistoryRoutes = require('./routes/chatHistory');

require('./config/passport');

const app = express();

app.use(compression());

app.use(express.json());
app.use(cookieParser());

app.use(cors({
  origin: process.env.FRONTEND_URL || 'https://custom-gpt-frontend-nine.vercel.app',
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'Accept', 'Origin', 'X-Requested-With'],
  exposedHeaders: ['Authorization']
}));

app.use(passport.initialize());


app.get("/", (req, res) => {
  res.status(200).send("API is running successfully!");
});

app.get("/health", (req, res) => {
  res.status(200).json({ status: "ok" });
});

app.use('/api/auth', authRoutes);
app.use('/api/auth', invitationRoutes);
app.use('/api/custom-gpts', customGptRoutes);
app.use('/api/chat-history', chatHistoryRoutes);

connectDB()
  .then(() => console.log('MongoDB connected at server startup'))
  .catch(err => console.error('Initial MongoDB connection failed:', err));

// Instead of app.listen, export the app
if (process.env.NODE_ENV !== 'production') {
  const PORT = process.env.PORT || 5000;
  app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
  });
}

// Export for serverless
module.exports = app;



