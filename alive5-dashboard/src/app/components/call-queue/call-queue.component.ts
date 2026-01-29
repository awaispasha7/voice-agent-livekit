/**
 * HITL Voice Handoff - Call Queue Component
 * Displays incoming calls awaiting human agent takeover
 */

import { Component, OnInit, OnDestroy } from '@angular/core';
import { Subscription } from 'rxjs';
import { Alive5SocketService } from '../../services/alive5-socket.service';
import { CallStateService } from '../../services/call-state.service';
import { IncomingCall } from '../../models/call.model';

@Component({
  selector: 'app-call-queue',
  templateUrl: './call-queue.component.html',
  styleUrls: ['./call-queue.component.css']
})
export class CallQueueComponent implements OnInit, OnDestroy {
  incomingCalls: IncomingCall[] = [];
  agentId: string = 'agent_' + Math.random().toString(36).substring(7);
  agentName: string = 'Human Agent';
  
  private subscriptions: Subscription[] = [];

  constructor(
    private alive5SocketService: Alive5SocketService,
    private callStateService: CallStateService
  ) {}

  ngOnInit(): void {
    // Subscribe to incoming calls
    const callsSub = this.alive5SocketService.incomingCalls$.subscribe(calls => {
      this.incomingCalls = calls;
    });
    this.subscriptions.push(callsSub);
  }

  ngOnDestroy(): void {
    this.subscriptions.forEach(sub => sub.unsubscribe());
  }

  /**
   * Accept an incoming call
   */
  async acceptCall(call: IncomingCall): Promise<void> {
    try {
      await this.callStateService.acceptCall(call, this.agentId, this.agentName);
    } catch (error) {
      console.error('Failed to accept call:', error);
      alert('Failed to accept call. Please try again.');
    }
  }

  /**
   * Reject an incoming call
   */
  async rejectCall(call: IncomingCall): Promise<void> {
    try {
      await this.callStateService.rejectCall(call, this.agentId, this.agentName);
    } catch (error) {
      console.error('Failed to reject call:', error);
      alert('Failed to reject call. Please try again.');
    }
  }

  /**
   * Format timestamp to readable time
   */
  formatTime(timestamp: number): string {
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
  }

  /**
   * Get time elapsed since call came in
   */
  getWaitingTime(timestamp: number): string {
    const now = Date.now();
    const elapsed = Math.floor((now - timestamp) / 1000);
    
    if (elapsed < 60) {
      return `${elapsed}s`;
    } else {
      const minutes = Math.floor(elapsed / 60);
      return `${minutes}m ${elapsed % 60}s`;
    }
  }
}

