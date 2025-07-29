const path = require('path');

module.exports = {
  entry: './src/index.js',
  mode: 'production',
  target: 'webworker',
  output: {
    filename: 'index.js',
    path: path.resolve(__dirname, 'dist'),
    library: {
      type: 'module'
    }
  },
  experiments: {
    outputModule: true
  },
  resolve: {
    extensions: ['.js', '.json']
  },
  optimization: {
    minimize: true
  }
};