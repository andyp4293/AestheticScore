const webpack = require('webpack');
const dotenv = require('dotenv-webpack');

module.exports = {
  plugins: [
    new dotenv()
  ]
}; 