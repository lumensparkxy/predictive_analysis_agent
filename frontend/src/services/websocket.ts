import { io, Socket } from 'socket.io-client';
import { API_CONFIG, WS_ENDPOINTS, DASHBOARD_CONFIG } from '../config/config';
import { RealTimeData } from '../types';

export type WebSocketEventHandler = (data: any) => void;

class WebSocketService {
  private socket: Socket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = DASHBOARD_CONFIG.MAX_RECONNECT_ATTEMPTS;
  private reconnectDelay = DASHBOARD_CONFIG.WEBSOCKET_RECONNECT_DELAY;
  private eventHandlers: Map<string, WebSocketEventHandler[]> = new Map();

  constructor() {
    this.initializeSocket();
  }

  private initializeSocket() {
    try {
      this.socket = io(API_CONFIG.WEBSOCKET_URL, {
        autoConnect: false,
        reconnection: true,
        reconnectionDelay: this.reconnectDelay,
        reconnectionAttempts: this.maxReconnectAttempts,
        timeout: 5000,
        transports: ['websocket'],
      });

      this.setupEventListeners();
    } catch (error) {
      console.error('Failed to initialize WebSocket:', error);
    }
  }

  private setupEventListeners() {
    if (!this.socket) return;

    // Connection events
    this.socket.on('connect', () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
      this.emit('connection_status', { connected: true });
    });

    this.socket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);
      this.emit('connection_status', { connected: false, reason });
    });

    this.socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      this.handleReconnect();
    });

    // Data events
    this.socket.on('real_time_update', (data: RealTimeData) => {
      this.emit('real_time_update', data);
    });

    this.socket.on('alert_notification', (data) => {
      this.emit('alert_notification', data);
    });

    this.socket.on('anomaly_notification', (data) => {
      this.emit('anomaly_notification', data);
    });

    // Generic message handler
    this.socket.on('message', (data) => {
      this.emit('message', data);
    });
  }

  private handleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      this.emit('connection_error', { 
        message: 'Unable to reconnect to server',
        maxAttemptsReached: true 
      });
      return;
    }

    this.reconnectAttempts++;
    console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
    
    setTimeout(() => {
      this.connect();
    }, this.reconnectDelay * this.reconnectAttempts);
  }

  // Public methods
  connect() {
    if (this.socket && !this.socket.connected) {
      this.socket.connect();
    }
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
    }
  }

  // Event handling
  on(event: string, handler: WebSocketEventHandler) {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, []);
    }
    this.eventHandlers.get(event)?.push(handler);
  }

  off(event: string, handler: WebSocketEventHandler) {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
  }

  private emit(event: string, data: any) {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(data);
        } catch (error) {
          console.error(`Error in WebSocket event handler for ${event}:`, error);
        }
      });
    }
  }

  // Send data
  send(event: string, data: any) {
    if (this.socket && this.socket.connected) {
      this.socket.emit(event, data);
    } else {
      console.warn('WebSocket not connected, cannot send data');
    }
  }

  // Status methods
  isConnected() {
    return this.socket?.connected || false;
  }

  getConnectionStatus() {
    return {
      connected: this.isConnected(),
      reconnectAttempts: this.reconnectAttempts,
      maxReconnectAttempts: this.maxReconnectAttempts,
    };
  }

  // Cleanup
  destroy() {
    this.eventHandlers.clear();
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }
}

// Real-time data service for dashboard
export class RealTimeDataService {
  private wsService: WebSocketService;
  private updateInterval: number;
  private isActive = false;
  private subscribers: Map<string, (data: any) => void> = new Map();

  constructor() {
    this.wsService = new WebSocketService();
    this.updateInterval = DASHBOARD_CONFIG.CHART_UPDATE_INTERVAL;
    this.setupSubscriptions();
  }

  private setupSubscriptions() {
    // Subscribe to real-time updates
    this.wsService.on('real_time_update', (data) => {
      this.notifySubscribers('metrics', data);
    });

    // Subscribe to alerts
    this.wsService.on('alert_notification', (data) => {
      this.notifySubscribers('alerts', data);
    });

    // Subscribe to anomalies
    this.wsService.on('anomaly_notification', (data) => {
      this.notifySubscribers('anomalies', data);
    });

    // Subscribe to connection status
    this.wsService.on('connection_status', (status) => {
      this.notifySubscribers('connection', status);
    });
  }

  private notifySubscribers(type: string, data: any) {
    this.subscribers.forEach((callback, key) => {
      if (key.startsWith(type)) {
        callback(data);
      }
    });
  }

  // Public methods
  start() {
    if (!this.isActive) {
      this.isActive = true;
      this.wsService.connect();
    }
  }

  stop() {
    if (this.isActive) {
      this.isActive = false;
      this.wsService.disconnect();
    }
  }

  // Subscribe to real-time data
  subscribe(type: 'metrics' | 'alerts' | 'anomalies' | 'connection', callback: (data: any) => void) {
    const key = `${type}_${Date.now()}_${Math.random()}`;
    this.subscribers.set(key, callback);
    return key;
  }

  // Unsubscribe from real-time data
  unsubscribe(subscriptionKey: string) {
    this.subscribers.delete(subscriptionKey);
  }

  // Send data
  sendData(event: string, data: any) {
    this.wsService.send(event, data);
  }

  // Status methods
  getStatus() {
    return {
      isActive: this.isActive,
      connection: this.wsService.getConnectionStatus(),
      subscribers: this.subscribers.size,
    };
  }

  // Cleanup
  destroy() {
    this.stop();
    this.subscribers.clear();
    this.wsService.destroy();
  }
}

// Singleton instance
export const realTimeService = new RealTimeDataService();
export default realTimeService;