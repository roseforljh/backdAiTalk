/**
 * Simple logger for Cloudflare Workers
 */

class Logger {
  constructor() {
    this.logLevel = 'INFO'; // Can be set from environment
  }

  setLevel(level) {
    this.logLevel = level.toUpperCase();
  }

  shouldLog(level) {
    const levels = { DEBUG: 0, INFO: 1, WARN: 2, ERROR: 3 };
    return levels[level] >= levels[this.logLevel];
  }

  formatMessage(level, message, ...args) {
    const timestamp = new Date().toISOString();
    const formattedArgs = args.length > 0 ? ' ' + args.map(arg => 
      typeof arg === 'object' ? JSON.stringify(arg) : String(arg)
    ).join(' ') : '';
    
    return `${timestamp} ${level.padEnd(5)} [EzTalkWorker] - ${message}${formattedArgs}`;
  }

  debug(message, ...args) {
    if (this.shouldLog('DEBUG')) {
      console.log(this.formatMessage('DEBUG', message, ...args));
    }
  }

  info(message, ...args) {
    if (this.shouldLog('INFO')) {
      console.log(this.formatMessage('INFO', message, ...args));
    }
  }

  warn(message, ...args) {
    if (this.shouldLog('WARN')) {
      console.warn(this.formatMessage('WARN', message, ...args));
    }
  }

  error(message, ...args) {
    if (this.shouldLog('ERROR')) {
      console.error(this.formatMessage('ERROR', message, ...args));
    }
  }
}

export const logger = new Logger();