const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const axios = require('axios');

const app = express();
const PORT = 3001; 

app.use(cors());
app.use(bodyParser.json());

app.get('/', (req, res) => {
  res.send(' MediScan Node.js Proxy Server is running ');
});

app.post('/predict-parkinson', async (req, res) => {
  try {
    const response = await axios.post('http://localhost:5000/predict-parkinson', req.body);
    res.status(200).json(response.data);
  } catch (error) {
    console.error(' Error forwarding to Flask:', error.message);
    res.status(500).json({ error: 'Failed to get prediction from Flask API' });
  }
});

app.post('/predict-diabetes', async (req, res) => {
  try {
    const response = await axios.post('http://localhost:5000/predict-diabetes', req.body);
    res.status(200).json(response.data);
  } catch (error) {
    console.error(' Error forwarding to Flask:', error.message);
    res.status(500).json({ error: 'Failed to get prediction from Flask API' });
  }
});

app.post('/predict-heart', async (req, res) => {
  try {
    const response = await axios.post('http://localhost:5000/predict-heart', req.body);
    res.status(200).json(response.data);
  } catch (error) {
    console.error(' Error forwarding to Flask:', error.message);
    res.status(500).json({ error: 'Failed to get prediction from Flask API' });
  }
});

app.listen(PORT, () => {
  console.log(` Node.js server running at http://localhost:${PORT}`);
});
