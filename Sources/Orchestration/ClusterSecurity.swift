import Foundation
import Network

// ── Cluster wire security (P0 red-team blocker: the control plane was unauthenticated plaintext TCP on
// all interfaces, and worker-returned "candidate code" is compiled+run on the coordinator — so any LAN
// host could inject executable content). Fix: wrap the TCP connection in TLS with a PRE-SHARED KEY.
//
// With a PSK (env SWIFTLM_CLUSTER_PSK, same on both boxes), the channel is encrypted AND authenticated:
// only peers holding the key complete the TLS handshake, so a random host can neither read traffic nor
// submit jobs / return crafted completions. Without a PSK we fall back to plain TCP (loopback/dev) with
// a loud warning — keeps the existing loopback tests and single-box runs working unchanged.

/// Build `NWParameters` for the cluster wire. `psk` nil/empty → plain TCP (warned); else TLS-PSK.
/// Public so the MLXBackend distributed-inference side-channel reuses the same PSK/plaintext policy.
public func clusterParameters(psk: String?) -> NWParameters {
    guard let psk, !psk.isEmpty else {
        clog("[cluster] ⚠️  no SWIFTLM_CLUSTER_PSK — wire is UNAUTHENTICATED plaintext TCP (loopback/dev only)")
        return NWParameters(tls: nil, tcp: NWProtocolTCP.Options())
    }
    let tls = NWProtocolTLS.Options()
    let key = Data(psk.utf8).withUnsafeBytes { DispatchData(bytes: $0) }
    let identity = Data("swiftlm-cluster".utf8).withUnsafeBytes { DispatchData(bytes: $0) }
    sec_protocol_options_add_pre_shared_key(tls.securityProtocolOptions,
                                            key as __DispatchData, identity as __DispatchData)
    // TLS 1.3 uses external PSKs natively; AES-128-GCM-SHA256 is the mandatory-to-implement suite.
    sec_protocol_options_append_tls_ciphersuite(tls.securityProtocolOptions,
                                                tls_ciphersuite_t.AES_128_GCM_SHA256)
    return NWParameters(tls: tls, tcp: NWProtocolTCP.Options())
}
