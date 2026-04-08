use async_graphql::futures_util::Stream;
use async_graphql::{Context, Result, Subscription};
use tokio::sync::broadcast;

use sphereql_core::{Contains, SphericalPoint};

use crate::types::{RegionInput, SphericalPointOutput};

#[derive(async_graphql::Enum, Copy, Clone, Eq, PartialEq, Debug)]
pub enum SpatialEventType {
    Entered,
    Left,
    Moved,
}

#[derive(async_graphql::SimpleObject, Debug, Clone)]
pub struct SpatialEvent {
    pub event_type: SpatialEventType,
    pub point: SphericalPointOutput,
    pub item_id: String,
}

pub struct SpatialEventBus {
    sender: broadcast::Sender<SpatialEvent>,
}

impl SpatialEventBus {
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self { sender }
    }

    pub fn publish(&self, event: SpatialEvent) {
        let _ = self.sender.send(event);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<SpatialEvent> {
        self.sender.subscribe()
    }
}

pub struct SphericalSubscriptionRoot;

#[Subscription]
impl SphericalSubscriptionRoot {
    async fn item_entered_region(
        &self,
        ctx: &Context<'_>,
        region: RegionInput,
    ) -> Result<impl Stream<Item = SpatialEvent>> {
        let bus = ctx.data::<SpatialEventBus>()?;
        let mut rx = bus.subscribe();
        let region = region.to_core()?;

        let stream = async_graphql::async_stream::stream! {
            while let Ok(event) = rx.recv().await {
                if event.event_type == SpatialEventType::Entered {
                    let core_point = SphericalPoint::new_unchecked(
                        event.point.r,
                        event.point.theta,
                        event.point.phi,
                    );
                    if region.contains(&core_point) {
                        yield event;
                    }
                }
            }
        };

        Ok(stream)
    }

    async fn item_left_region(
        &self,
        ctx: &Context<'_>,
        region: RegionInput,
    ) -> Result<impl Stream<Item = SpatialEvent>> {
        let bus = ctx.data::<SpatialEventBus>()?;
        let mut rx = bus.subscribe();
        let region = region.to_core()?;

        let stream = async_graphql::async_stream::stream! {
            while let Ok(event) = rx.recv().await {
                if event.event_type == SpatialEventType::Left {
                    let core_point = SphericalPoint::new_unchecked(
                        event.point.r,
                        event.point.theta,
                        event.point.phi,
                    );
                    if region.contains(&core_point) {
                        yield event;
                    }
                }
            }
        };

        Ok(stream)
    }

    async fn spatial_events(&self, ctx: &Context<'_>) -> Result<impl Stream<Item = SpatialEvent>> {
        let bus = ctx.data::<SpatialEventBus>()?;
        let mut rx = bus.subscribe();

        let stream = async_graphql::async_stream::stream! {
            while let Ok(event) = rx.recv().await {
                yield event;
            }
        };

        Ok(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_4;

    fn make_event(event_type: SpatialEventType, r: f64, theta: f64, phi: f64) -> SpatialEvent {
        SpatialEvent {
            event_type,
            point: SphericalPointOutput {
                r,
                theta,
                phi,
                theta_degrees: theta.to_degrees(),
                phi_degrees: phi.to_degrees(),
            },
            item_id: format!("item-{r}-{theta}-{phi}"),
        }
    }

    #[tokio::test]
    async fn event_bus_publish_subscribe() {
        let bus = SpatialEventBus::new(16);
        let mut rx = bus.subscribe();

        let event = make_event(SpatialEventType::Entered, 1.0, 0.5, FRAC_PI_4);
        bus.publish(event.clone());

        let received = rx.recv().await.unwrap();
        assert_eq!(received.item_id, "item-1-0.5-0.7853981633974483");
        assert_eq!(received.event_type, SpatialEventType::Entered);
        assert!((received.point.r - 1.0).abs() < 1e-12);
    }

    #[tokio::test]
    async fn multiple_subscribers_receive_events() {
        let bus = SpatialEventBus::new(16);
        let mut rx1 = bus.subscribe();
        let mut rx2 = bus.subscribe();

        let event = make_event(SpatialEventType::Moved, 2.0, 1.0, 0.5);
        bus.publish(event.clone());

        let r1 = rx1.recv().await.unwrap();
        let r2 = rx2.recv().await.unwrap();

        assert_eq!(r1.item_id, r2.item_id);
        assert_eq!(r1.event_type, SpatialEventType::Moved);
        assert_eq!(r2.event_type, SpatialEventType::Moved);
    }

    #[tokio::test]
    async fn event_type_filtering() {
        let bus = SpatialEventBus::new(16);
        let mut rx = bus.subscribe();

        bus.publish(make_event(SpatialEventType::Entered, 1.0, 0.5, 0.5));
        bus.publish(make_event(SpatialEventType::Left, 1.0, 0.6, 0.6));
        bus.publish(make_event(SpatialEventType::Moved, 1.0, 0.7, 0.7));
        bus.publish(make_event(SpatialEventType::Entered, 2.0, 0.8, 0.8));

        let mut entered = Vec::new();
        for _ in 0..4 {
            let event = rx.recv().await.unwrap();
            if event.event_type == SpatialEventType::Entered {
                entered.push(event);
            }
        }

        assert_eq!(entered.len(), 2);
        assert!((entered[0].point.r - 1.0).abs() < 1e-12);
        assert!((entered[1].point.r - 2.0).abs() < 1e-12);
    }
}
