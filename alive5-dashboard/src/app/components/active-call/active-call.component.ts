/**
 * HITL Voice Handoff - Active Call Component
 * Displays and controls the currently active call
 */

import { Component, OnInit, OnDestroy } from '@angular/core';
import { Subscription } from 'rxjs';
import { CallStateService } from '../../services/call-state.service';
import { LiveKitService } from '../../services/livekit.service';
import { CallSession } from '../../models/call.model';

@Component({
  selector: 'app-active-call',
  templateUrl: './active-call.component.html',
  styleUrls: ['./active-call.component.css']
})
export class ActiveCallComponent implements OnInit, OnDestroy {
  activeCall: CallSession | null = null;
  callDuration: number = 0;
  isMuted: boolean = false;
  isConnected: boolean = false;
  
  private subscriptions: Subscription[] = [];

  constructor(
    private callStateService: CallStateService,
    private liveKitService: LiveKitService
  ) {}

  ngOnInit(): void {
    // Subscribe to active call
    const callSub = this.callStateService.activeCall$.subscribe(call => {
      this.activeCall = call;
    });
    this.subscriptions.push(callSub);
    
    // Subscribe to call duration
    const durationSub = this.callStateService.callDuration$.subscribe(duration => {
      this.callDuration = duration;
    });
    this.subscriptions.push(durationSub);
    
    // Subscribe to mute state
    const muteSub = this.liveKitService.muted$.subscribe(muted => {
      this.isMuted = muted;
    });
    this.subscriptions.push(muteSub);
    
    // Subscribe to connection state
    const connectedSub = this.liveKitService.connected$.subscribe(connected => {
      this.isConnected = connected;
    });
    this.subscriptions.push(connectedSub);
  }

  ngOnDestroy(): void {
    this.subscriptions.forEach(sub => sub.unsubscribe());
  }

  /**
   * Toggle microphone mute
   */
  async toggleMute(): Promise<void> {
    try {
      await this.liveKitService.toggleMute();
    } catch (error) {
      console.error('Failed to toggle mute:', error);
    }
  }

  /**
   * End the call
   */
  async endCall(): Promise<void> {
    if (confirm('Are you sure you want to end this call?')) {
      try {
        await this.callStateService.endCall(false);
      } catch (error) {
        console.error('Failed to end call:', error);
        alert('Failed to end call. Please try again.');
      }
    }
  }

  /**
   * End call and resume AI
   */
  async endCallResumeAI(): Promise<void> {
    if (confirm('End call and transfer back to AI agent?')) {
      try {
        await this.callStateService.endCall(true);
      } catch (error) {
        console.error('Failed to end call:', error);
        alert('Failed to end call. Please try again.');
      }
    }
  }

  /**
   * Format call duration to MM:SS
   */
  formatDuration(seconds: number): string {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }

  /**
   * Get current date for display
   */
  getCurrentDate(): string {
    const now = new Date();
    return now.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  }
}

