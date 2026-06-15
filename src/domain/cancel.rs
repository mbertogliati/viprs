use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::domain::error::ViprsError;

/// A cooperative cancellation token for pipeline execution.
///
/// Servers create one token per request and pass it to the scheduler. Schedulers poll this token
/// between tiles so request aborts can stop work without injecting shared mutable state into pixel
/// kernels.
///
/// # Examples
/// ```rust
/// # use viprs::domain::cancel::CancellationToken;
/// let token = CancellationToken::new();
/// assert!(!token.is_cancelled());
/// ```
#[derive(Clone, Debug)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Create a fresh token in the active state.
    ///
    /// This gives one execution request a cancel signal that can be cloned across components.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs::domain::cancel::CancellationToken;
    /// let token = CancellationToken::new();
    /// assert!(!token.is_cancelled());
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Mark the token as cancelled for all clones.
    ///
    /// Schedulers observe this flag and stop after their current tile boundary.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs::domain::cancel::CancellationToken;
    /// let token = CancellationToken::new();
    /// token.cancel();
    /// assert!(token.is_cancelled());
    /// ```
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    /// Return whether cancellation has been requested.
    ///
    /// This is the cheap polling hook used by schedulers between units of work.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs::domain::cancel::CancellationToken;
    /// let token = CancellationToken::new();
    /// assert!(!token.is_cancelled());
    /// ```
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// Check if cancelled, returning `Err(ViprsError::Cancelled)` if so.
    pub fn check_cancelled(&self) -> Result<(), ViprsError> {
        if self.is_cancelled() {
            return Err(ViprsError::Cancelled);
        }
        Ok(())
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::CancellationToken;

    #[test]
    fn fresh_token_is_not_cancelled() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn cancel_sets_the_flag() {
        let token = CancellationToken::new();
        token.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn clone_shares_state() {
        let token = CancellationToken::new();
        let clone = token.clone();

        clone.cancel();

        assert!(token.is_cancelled());
        assert!(clone.is_cancelled());
    }

    #[test]
    fn default_creates_active_token() {
        let token = CancellationToken::default();
        assert!(!token.is_cancelled());
    }
}
