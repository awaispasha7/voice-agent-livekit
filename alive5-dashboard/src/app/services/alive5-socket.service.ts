/**
 * HITL Voice Handoff - Alive5 Socket Service
 * Handles Socket.IO connection to Alive5 for incoming call notifications
 */

import { Injectable } from '@angular/core';
import { io, Socket } from 'socket.io-client';
import { BehaviorSubject, Observable } from 'rxjs';
import { IncomingCall } from '../models/call.model';

@Injectable({
  providedIn: 'root'
})
export class Alive5SocketService {
  private socket: Socket | null = null;
  private connectedSubject = new BehaviorSubject<boolean>(false);
  public connected$: Observable<boolean> = this.connectedSubject.asObservable();
  
  private incomingCallsSubject = new BehaviorSubject<IncomingCall[]>([]);
  public incomingCalls$: Observable<IncomingCall[]> = this.incomingCallsSubject.asObservable();

  constructor() {}

  /**
   * Connect to backend Socket.IO server (for HITL dashboard)
   */
  connect(apiKey: string, agentId: string, socketUrl: string = 'http://localhost:8000'): void {
    try {
      console.log('[Alive5Socket] Connecting to backend:', socketUrl);
      
      // Create socket connection to our backend (no auth needed for dashboard)
      this.socket = io(socketUrl, {
        transports: ['websocket'],
        path: '/socket.io',
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionAttempts: 5
      });

      // Set up event listeners
      this.setupEventListeners();
    } catch (error) {
      console.error('[Alive5Socket] Failed to connect:', error);
      throw error;
    }
  }

  /**
   * Set up socket event listeners
   */
  private setupEventListeners(): void {
    if (!this.socket) return;

    this.socket.on('connect', () => {
      console.log('[Alive5Socket] Connected');
      this.connectedSubject.next(true);
    });

    this.socket.on('disconnect', () => {
      console.log('[Alive5Socket] Disconnected');
      this.connectedSubject.next(false);
    });

    this.socket.on('incoming_human_call', (data: IncomingCall) => {
      console.log('[Alive5Socket] Incoming call:', data);
      const currentCalls = this.incomingCallsSubject.value;
      this.incomingCallsSubject.next([...currentCalls, data]);
    });

    this.socket.on('error', (error: any) => {
      console.error('[Alive5Socket] Socket error:', error);
    });
  }

  /**
   * Remove a call from the incoming queue (when accepted or rejected)
   */
  removeIncomingCall(roomName: string): void {
    const currentCalls = this.incomingCallsSubject.value;
    const filteredCalls = currentCalls.filter(call => call.room_name !== roomName);
    this.incomingCallsSubject.next(filteredCalls);
  }

  /**
   * Disconnect from socket
   */
  disconnect(): void {
    if (this.socket) {
      console.log('[Alive5Socket] Disconnecting');
      this.socket.disconnect();
      this.socket = null;
      this.connectedSubject.next(false);
    }
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.socket?.connected ?? false;
  }
}

