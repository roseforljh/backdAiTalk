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
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: [
              ['@babel/preset-env', {
                targets: {
                  browsers: ['last 2 Chrome versions']
                }
              }]
            ]
          }
        }
      }
    ]
  },
  optimization: {
    minimize: true
  }
};