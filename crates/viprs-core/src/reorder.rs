//! DAG reordering helpers for minimizing peak tile and buffer usage.
//!
//! These utilities score valid topological orders so compiled pipelines can execute with lower
//! intermediate tile pressure.

use crate::op::NodeSpec;

/// Stable identifier for a node inside a reordering graph.
///
/// The planner uses slice indices so it can operate on compact graph metadata without owning the
/// compiled nodes themselves.
///
/// # Examples
/// ```rust
/// # use viprs::domain::reorder::ReorderNodeId;
/// let node: ReorderNodeId = 0;
/// assert_eq!(node, 0);
/// ```
pub type ReorderNodeId = usize;

/// Per-node metadata used by DAG reordering heuristics.
///
/// This captures whether a node keeps an output tile alive after execution, which directly affects
/// peak buffer pressure.
///
/// # Examples
/// ```rust
/// # use viprs::domain::{op::NodeSpec, reorder::ReorderNode};
/// let node = ReorderNode::transform(NodeSpec::identity(8, 8));
/// assert!(node.retains_output_tile);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReorderNode {
    /// Stores the `spec` value for this item.
    pub spec: NodeSpec,
    /// Stores the `retains_output_tile` value for this item.
    pub retains_output_tile: bool,
}

impl ReorderNode {
    /// Create metadata for a transform node that retains its own output tile.
    ///
    /// Transform nodes materialize new pixels, so their output remains live until all consumers run.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs::domain::{op::NodeSpec, reorder::ReorderNode};
    /// let node = ReorderNode::transform(NodeSpec::identity(4, 4));
    /// assert!(node.retains_output_tile);
    /// ```
    #[must_use]
    pub const fn transform(spec: NodeSpec) -> Self {
        Self {
            spec,
            retains_output_tile: true,
        }
    }

    /// Create metadata for a zero-copy view node.
    ///
    /// View nodes reuse an upstream tile instead of allocating a new output tile.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs::domain::{op::NodeSpec, reorder::ReorderNode};
    /// let node = ReorderNode::view(NodeSpec::identity(4, 4));
    /// assert!(!node.retains_output_tile);
    /// ```
    #[must_use]
    pub const fn view(spec: NodeSpec) -> Self {
        Self {
            spec,
            retains_output_tile: false,
        }
    }
}

/// Result of reordering a DAG under tile-liveness heuristics.
///
/// The plan stores the execution order plus the estimated peak number of simultaneously live tiles.
///
/// # Examples
/// ```rust
/// # use viprs::domain::reorder::ReorderPlan;
/// let plan = ReorderPlan { order: vec![0, 1], peak_live_tiles: 1 };
/// assert_eq!(plan.peak_live_tiles, 1);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReorderPlan {
    /// Butterworth order associated with this mask profile.
    pub order: Vec<ReorderNodeId>,
    /// Stores the `peak_live_tiles` value for this item.
    pub peak_live_tiles: usize,
}

/// Errors raised while validating or ordering a reordering graph.
///
/// These distinguish invalid node references from dependency cycles.
///
/// # Examples
/// ```rust
/// # use viprs::domain::reorder::ReorderError;
/// assert!(matches!(ReorderError::Cycle, ReorderError::Cycle));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReorderError {
    /// Returned when an `InvalidNode` condition is detected.
    InvalidNode {
        /// Pipeline node index associated with this condition.
        node: ReorderNodeId,
        /// Stores the `node_count` value for this item.
        node_count: usize,
    },
    /// Returned when `Cycle` applies.
    Cycle,
}

/// Return a stable topological order for the provided DAG.
///
/// This preserves deterministic ordering among ready nodes while rejecting invalid edges or cycles.
///
/// # Examples
/// ```rust
/// # use viprs::domain::{op::NodeSpec, reorder::{stable_topological_order, ReorderNode}};
/// let nodes = [ReorderNode::transform(NodeSpec::identity(1, 1))];
/// let order = stable_topological_order(&nodes, &[]).unwrap();
/// assert_eq!(order, vec![0]);
/// ```
pub fn stable_topological_order(
    nodes: &[ReorderNode],
    edges: &[(ReorderNodeId, ReorderNodeId)],
) -> Result<Vec<ReorderNodeId>, ReorderError> {
    let adjacency = build_adjacency(nodes.len(), edges)?;
    let mut indegree = vec![0usize; nodes.len()];
    for downstreams in &adjacency {
        for &downstream in downstreams {
            indegree[downstream] += 1;
        }
    }

    let mut ready: Vec<ReorderNodeId> =
        (0..nodes.len()).filter(|&idx| indegree[idx] == 0).collect();
    let mut order = Vec::with_capacity(nodes.len());

    while let Some(node) = ready.first().copied() {
        ready.remove(0);
        order.push(node);
        for &downstream in &adjacency[node] {
            indegree[downstream] -= 1;
            if indegree[downstream] == 0 {
                ready.push(downstream);
            }
        }
    }

    if order.len() != nodes.len() {
        return Err(ReorderError::Cycle);
    }

    Ok(order)
}

/// Estimate the peak number of simultaneously live tiles for a specific node order.
///
/// This helps compare candidate execution orders before allocating scheduler buffers.
///
/// # Examples
/// ```rust
/// # use viprs::domain::{op::NodeSpec, reorder::{peak_live_tiles_for_order, ReorderNode}};
/// let nodes = [ReorderNode::transform(NodeSpec::identity(1, 1))];
/// assert_eq!(peak_live_tiles_for_order(&nodes, &[], &[0]).unwrap(), 1);
/// ```
pub fn peak_live_tiles_for_order(
    nodes: &[ReorderNode],
    edges: &[(ReorderNodeId, ReorderNodeId)],
    order: &[ReorderNodeId],
) -> Result<usize, ReorderError> {
    let adjacency = build_adjacency(nodes.len(), edges)?;
    let predecessors = build_predecessors(nodes.len(), edges)?;
    let mut remaining_consumers: Vec<usize> = adjacency.iter().map(Vec::len).collect();
    let mut live_tiles = 0usize;
    let mut peak_live_tiles = 0usize;

    for &node in order {
        if node >= nodes.len() {
            return Err(ReorderError::InvalidNode {
                node,
                node_count: nodes.len(),
            });
        }

        for &upstream in &predecessors[node] {
            if remaining_consumers[upstream] == 0 {
                continue;
            }
            remaining_consumers[upstream] -= 1;
            if remaining_consumers[upstream] == 0 && nodes[upstream].retains_output_tile {
                live_tiles -= 1;
            }
        }

        if nodes[node].retains_output_tile {
            live_tiles += 1;
            peak_live_tiles = peak_live_tiles.max(live_tiles);
        }
    }

    Ok(peak_live_tiles)
}

/// Estimate the number of buffer slots required for a specific execution order.
///
/// This models tile-buffer reuse across transform and view nodes without running the pipeline.
///
/// # Examples
/// ```rust
/// # use viprs::domain::{op::NodeSpec, reorder::{peak_buffer_slots_for_order, ReorderNode}};
/// let nodes = [ReorderNode::transform(NodeSpec::identity(1, 1))];
/// assert!(peak_buffer_slots_for_order(&nodes, &[], &[0]).unwrap() >= 1);
/// ```
pub fn peak_buffer_slots_for_order(
    nodes: &[ReorderNode],
    edges: &[(ReorderNodeId, ReorderNodeId)],
    order: &[ReorderNodeId],
) -> Result<usize, ReorderError> {
    let predecessors = build_predecessors(nodes.len(), edges)?;
    let mut buffer_owner = vec![None; nodes.len()];
    let mut output_buffers = vec![None; nodes.len()];

    for &node in order {
        if node >= nodes.len() {
            return Err(ReorderError::InvalidNode {
                node,
                node_count: nodes.len(),
            });
        }

        if nodes[node].retains_output_tile {
            buffer_owner[node] = Some(node);
            continue;
        }

        buffer_owner[node] = predecessors[node]
            .first()
            .and_then(|&upstream| buffer_owner[upstream]);
    }

    let mut remaining_buffer_consumers = vec![0usize; nodes.len()];
    for &node in order {
        if !nodes[node].retains_output_tile {
            continue;
        }
        for &upstream in &predecessors[node] {
            if let Some(owner) = buffer_owner[upstream] {
                remaining_buffer_consumers[owner] += 1;
            }
        }
    }

    let mut free_buffers = Vec::new();
    let mut max_buffer = 0usize;
    let mut peak_buffers = 1usize;

    for &node in order {
        let input_buffers: Vec<usize> = predecessors[node]
            .iter()
            .map(|&upstream| output_buffers[upstream].unwrap_or(0))
            .collect();

        if nodes[node].retains_output_tile {
            let max_input = input_buffers.iter().copied().max().unwrap_or(0);
            let reuse_pos = free_buffers
                .iter()
                .enumerate()
                .filter(|(_, idx)| **idx > max_input)
                .min_by_key(|(_, idx)| **idx)
                .map(|(pos, _)| pos);
            let output_buffer = reuse_pos.map_or_else(
                || {
                    max_buffer += 1;
                    max_buffer
                },
                |pos| free_buffers.swap_remove(pos),
            );
            output_buffers[node] = Some(output_buffer);
            peak_buffers = peak_buffers.max(max_buffer + 1);
        } else {
            output_buffers[node] = input_buffers.first().copied().or(Some(0));
        }

        for &upstream in &predecessors[node] {
            let Some(owner) = buffer_owner[upstream] else {
                continue;
            };
            if remaining_buffer_consumers[owner] == 0 {
                continue;
            }
            remaining_buffer_consumers[owner] -= 1;
            if remaining_buffer_consumers[owner] == 0
                && let Some(buffer) = output_buffers[owner]
            {
                free_buffers.push(buffer);
            }
        }
    }

    Ok(peak_buffers)
}

/// Reorder a DAG to reduce live tile pressure while preserving dependencies.
///
/// This returns a heuristic execution plan along with its estimated peak live-tile count.
///
/// # Examples
/// ```rust
/// # use viprs::domain::{op::NodeSpec, reorder::{reorder_dag, ReorderNode}};
/// let nodes = [ReorderNode::transform(NodeSpec::identity(1, 1))];
/// let plan = reorder_dag(&nodes, &[]).unwrap();
/// assert_eq!(plan.order, vec![0]);
/// ```
pub fn reorder_dag(
    nodes: &[ReorderNode],
    edges: &[(ReorderNodeId, ReorderNodeId)],
) -> Result<ReorderPlan, ReorderError> {
    let adjacency = build_adjacency(nodes.len(), edges)?;
    let predecessors = build_predecessors(nodes.len(), edges)?;
    let descendant_counts = descendant_counts(&adjacency);
    let mut indegree = vec![0usize; nodes.len()];
    for downstreams in &adjacency {
        for &downstream in downstreams {
            indegree[downstream] += 1;
        }
    }

    let mut remaining_consumers: Vec<usize> = adjacency.iter().map(Vec::len).collect();
    let mut ready: Vec<ReorderNodeId> =
        (0..nodes.len()).filter(|&idx| indegree[idx] == 0).collect();
    let mut order = Vec::with_capacity(nodes.len());
    let mut live_tiles = 0usize;
    let mut peak_live_tiles = 0usize;

    while !ready.is_empty() {
        let mut best_pos = 0usize;
        let mut best_score = candidate_score(
            ready[0],
            nodes,
            &adjacency,
            &predecessors,
            &remaining_consumers,
            live_tiles,
            descendant_counts[ready[0]],
        );

        for (pos, &candidate) in ready.iter().enumerate().skip(1) {
            let score = candidate_score(
                candidate,
                nodes,
                &adjacency,
                &predecessors,
                &remaining_consumers,
                live_tiles,
                descendant_counts[candidate],
            );
            if score < best_score {
                best_pos = pos;
                best_score = score;
            }
        }

        let node = ready.remove(best_pos);
        order.push(node);

        for &upstream in &predecessors[node] {
            if remaining_consumers[upstream] == 0 {
                continue;
            }
            remaining_consumers[upstream] -= 1;
            if remaining_consumers[upstream] == 0 && nodes[upstream].retains_output_tile {
                live_tiles -= 1;
            }
        }

        if nodes[node].retains_output_tile {
            live_tiles += 1;
            peak_live_tiles = peak_live_tiles.max(live_tiles);
        }

        for &downstream in &adjacency[node] {
            indegree[downstream] -= 1;
            if indegree[downstream] == 0 {
                ready.push(downstream);
            }
        }
    }

    if order.len() != nodes.len() {
        return Err(ReorderError::Cycle);
    }

    Ok(ReorderPlan {
        order,
        peak_live_tiles,
    })
}

fn candidate_score(
    node: ReorderNodeId,
    nodes: &[ReorderNode],
    adjacency: &[Vec<ReorderNodeId>],
    predecessors: &[Vec<ReorderNodeId>],
    remaining_consumers: &[usize],
    live_tiles: usize,
    descendant_count: usize,
) -> (usize, usize, usize, ReorderNodeId) {
    let releases = predecessors[node]
        .iter()
        .filter(|&&upstream| {
            remaining_consumers[upstream] == 1 && nodes[upstream].retains_output_tile
        })
        .count();
    let predicted_live_tiles = live_tiles + usize::from(nodes[node].retains_output_tile) - releases;
    (
        predicted_live_tiles,
        descendant_count,
        adjacency[node].len(),
        node,
    )
}

fn build_adjacency(
    node_count: usize,
    edges: &[(ReorderNodeId, ReorderNodeId)],
) -> Result<Vec<Vec<ReorderNodeId>>, ReorderError> {
    let mut adjacency = vec![Vec::new(); node_count];
    for &(upstream, downstream) in edges {
        if upstream >= node_count {
            return Err(ReorderError::InvalidNode {
                node: upstream,
                node_count,
            });
        }
        if downstream >= node_count {
            return Err(ReorderError::InvalidNode {
                node: downstream,
                node_count,
            });
        }
        adjacency[upstream].push(downstream);
    }
    Ok(adjacency)
}

fn build_predecessors(
    node_count: usize,
    edges: &[(ReorderNodeId, ReorderNodeId)],
) -> Result<Vec<Vec<ReorderNodeId>>, ReorderError> {
    let mut predecessors = vec![Vec::new(); node_count];
    for &(upstream, downstream) in edges {
        if upstream >= node_count {
            return Err(ReorderError::InvalidNode {
                node: upstream,
                node_count,
            });
        }
        if downstream >= node_count {
            return Err(ReorderError::InvalidNode {
                node: downstream,
                node_count,
            });
        }
        predecessors[downstream].push(upstream);
    }
    Ok(predecessors)
}

fn descendant_counts(adjacency: &[Vec<ReorderNodeId>]) -> Vec<usize> {
    fn visit(
        node: ReorderNodeId,
        adjacency: &[Vec<ReorderNodeId>],
        memo: &mut [Option<usize>],
    ) -> usize {
        if let Some(count) = memo[node] {
            return count;
        }
        let count = adjacency[node]
            .iter()
            .map(|&child| 1 + visit(child, adjacency, memo))
            .sum();
        memo[node] = Some(count);
        count
    }

    let mut memo = vec![None; adjacency.len()];
    (0..adjacency.len())
        .map(|node| visit(node, adjacency, &mut memo))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transform() -> ReorderNode {
        ReorderNode::transform(NodeSpec::identity(128, 128))
    }

    #[test]
    fn linear_pipeline_keeps_stable_order() {
        let nodes = vec![transform(), transform(), transform()];
        let edges = vec![(0, 1), (1, 2)];

        let stable = stable_topological_order(&nodes, &edges).unwrap();
        let reordered = reorder_dag(&nodes, &edges).unwrap();

        assert_eq!(stable, vec![0, 1, 2]);
        assert_eq!(reordered.order, stable);
        assert_eq!(
            reordered.peak_live_tiles,
            peak_live_tiles_for_order(&nodes, &edges, &stable).unwrap()
        );
    }

    #[test]
    fn diamond_pipeline_finishes_one_branch_before_the_other() {
        let nodes = vec![
            transform(),
            transform(),
            transform(),
            transform(),
            transform(),
            transform(),
        ];
        let edges = vec![(0, 1), (1, 2), (0, 3), (3, 4), (2, 5), (4, 5)];

        let stable = stable_topological_order(&nodes, &edges).unwrap();
        let reordered = reorder_dag(&nodes, &edges).unwrap();

        assert_eq!(stable, vec![0, 1, 3, 2, 4, 5]);
        assert_eq!(reordered.order, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(reordered.peak_live_tiles, 2);
    }

    #[test]
    fn three_way_fanout_reduces_peak_live_tiles() {
        let nodes = vec![
            transform(),
            transform(),
            transform(),
            transform(),
            transform(),
            transform(),
            transform(),
            transform(),
            transform(),
            transform(),
            transform(),
        ];
        let edges = vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (0, 4),
            (4, 5),
            (5, 6),
            (0, 7),
            (7, 8),
            (8, 9),
            (3, 10),
            (6, 10),
            (9, 10),
        ];

        let stable = stable_topological_order(&nodes, &edges).unwrap();
        let reordered = reorder_dag(&nodes, &edges).unwrap();

        assert_eq!(stable, vec![0, 1, 4, 7, 2, 5, 8, 3, 6, 9, 10]);
        assert_eq!(reordered.order, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        assert!(
            peak_buffer_slots_for_order(&nodes, &edges, &reordered.order).unwrap()
                < peak_buffer_slots_for_order(&nodes, &edges, &stable).unwrap()
        );
    }
}
